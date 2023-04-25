import os
from math import ceil, sqrt
from typing import Any, Optional, Union, Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx, degree
from ml_collections import ConfigDict
from torch_geometric.data import Batch, Data

from data.data_utils import AttributedDataLoader, IsBetter, scale_grad, batched_edge_index_to_batched_adj, self_defined_softmax, MyPlateau
from data.metrics import get_eval
from imle.noise import GumbelDistribution
from imle.target import TargetDistribution
from imle.wrapper import imle
from training.aux_loss import get_degree_regularization, get_variance_regularization, get_original_bias
from training.gumbel_scheme import GumbelSampler
from training.imle_scheme import IMLEScheme
from training.simple_scheme import EdgeSIMPLEBatched

LARGE_NUMBER = 1.e10
Optimizer = Union[torch.optim.Adam,
                  torch.optim.SGD]
Scheduler = Union[torch.optim.lr_scheduler.ReduceLROnPlateau,
                  torch.optim.lr_scheduler.MultiStepLR]
Emb_model = Any
Train_model = Any
Loss = torch.nn.modules.loss


class Trainer:
    def __init__(self,
                 dataset: str,
                 task_type: str,
                 max_patience: int,
                 patience_target: str,
                 criterion: Loss,
                 device: Union[str, torch.device],
                 imle_configs: ConfigDict,
                 sample_configs: ConfigDict,
                 auxloss: ConfigDict,
                 wandb: Optional[Any] = None,
                 use_wandb: bool = False,
                 plot_args: Optional[ConfigDict] = None):
        super(Trainer, self).__init__()

        self.dataset = dataset
        self.task_type = task_type
        self.metric_comparator = IsBetter(self.task_type)
        self.criterion = criterion
        self.device = device

        self.sample_configs = sample_configs
        self.imle_configs = imle_configs

        self.best_val_loss = 1e5
        self.best_train_loss = 1e5
        self.best_val_metric = None
        self.best_train_metric = None
        self.patience = 0
        self.max_patience = max_patience
        self.patience_target = patience_target
        self.auxloss = auxloss

        self.wandb = wandb
        self.use_wandb = use_wandb
        self.plot_args = plot_args

        self.epoch = None

        self.sample_policy = sample_configs.sample_policy
        self.include_original_graph = sample_configs.include_original_graph
        if imle_configs is not None:
            # upstream may have micro batching
            self.micro_batch_embd = imle_configs.micro_batch_embd

            if imle_configs.sampler == 'imle':
                assert imle_configs.num_val_ensemble == 1
                imle_scheduler = IMLEScheme(sample_configs.sample_policy, sample_configs.sample_k)

                @imle(target_distribution=TargetDistribution(alpha=1.0, beta=imle_configs.beta),
                      noise_distribution=GumbelDistribution(0., imle_configs.noise_scale, self.device),
                      input_noise_temperature=1.,
                      target_noise_temperature=1.,)
                def imle_sample_scheme(logits: torch.Tensor):
                    return imle_scheduler.torch_sample_scheme(logits)

                # we perturb during training, but not validation
                self.train_forward = imle_sample_scheme
                self.val_forward = imle_scheduler.torch_sample_scheme
                self.sampler_class = imle_scheduler
            elif imle_configs.sampler == 'gumbel':
                assert imle_configs.num_val_ensemble == 1
                gumbel_sampler = GumbelSampler(sample_configs.sample_k, tau=imle_configs.tau, policy=sample_configs.sample_policy)
                self.train_forward = gumbel_sampler
                self.val_forward = gumbel_sampler.validation
                self.sampler_class = gumbel_sampler
            elif imle_configs.sampler == 'simple':
                simple_sampler = EdgeSIMPLEBatched(sample_configs.sample_k,
                                                   device,
                                                   val_ensemble=imle_configs.num_val_ensemble,
                                                   policy=sample_configs.sample_policy,
                                                   logits_activation=imle_configs.logits_activation)
                self.train_forward = simple_sampler
                self.val_forward = simple_sampler.validation
                self.sampler_class = simple_sampler
            else:
                raise ValueError

        if sample_configs.sample_policy is None:
            self.construct_duplicate_data = lambda x, *args: (x, None, None)
        elif imle_configs is None:
            self.construct_duplicate_data = lambda x, *args: (x[0], None, None)
        elif imle_configs is not None:
            self.construct_duplicate_data = self.diffable_rewire


    def check_datatype(self, data):
        if isinstance(data, Data):
            return data.to(self.device)
        elif isinstance(data, tuple) and isinstance(data[0], Data) and isinstance(data[1], list):
            return data[0].to(self.device), [g.to(self.device) for g in data[1]]
        else:
            raise TypeError(f"Unexpected dtype {type(data)}")


    def diffable_rewire(self, collate_data: Tuple[Data, List[Data]], emb_model: Emb_model, ):

        dat_batch, graphs = collate_data

        train = emb_model.training
        output_logits, real_node_node_mask = emb_model(dat_batch)

        if self.sample_policy == 'global_topk_semi' or (train and self.auxloss is not None and self.auxloss.origin_bias > 0.):
            # need to compute the dense adj matrix
            adj = batched_edge_index_to_batched_adj(dat_batch, torch.float)
            self.sampler_class.adj = adj

        padding_bias = (~real_node_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
        logits = output_logits - padding_bias

        auxloss = 0.
        if train and self.auxloss is not None:
            if self.auxloss.degree > 0:
                raise NotImplementedError
                # auxloss = auxloss + get_degree_regularization(node_mask, self.auxloss.degree, real_node_node_mask)
            if self.auxloss.variance > 0:
                auxloss = auxloss + get_variance_regularization(logits,
                                                                self.auxloss.variance,
                                                                real_node_node_mask)
            if self.auxloss.origin_bias > 0.:
                auxloss = auxloss + get_original_bias(adj, logits,
                                                      self.auxloss.origin_bias,
                                                      real_node_node_mask)

        # (#sampled, B, N, N, E), (B, N, N, E)
        node_mask, marginals = self.train_forward(logits) if train else self.val_forward(logits)
        VE, B, N, _, E = node_mask.shape

        if self.imle_configs.weight_edges == 'logits':
            # (#sampled, B, N, N, E)
            sampled_edge_weights = torch.vmap(
                torch.vmap(
                    torch.vmap(
                        self_defined_softmax,
                        in_dims=(None, 0),
                        out_dims=0),
                    in_dims=0, out_dims=0),
                in_dims=-1,
                out_dims=-1)(logits, node_mask)
        elif self.imle_configs.weight_edges == 'marginals':
            assert self.imle_configs.sampler == 'simple'
            # Maybe we should also try this with softmax?
            assert marginals is not None
            sampled_edge_weights = marginals[None].repeat(node_mask.shape[0], 1, 1, 1, 1)
        elif self.imle_configs.weight_edges == 'None' or self.imle_configs.weight_edges is None:
            sampled_edge_weights = node_mask
        else:
            raise ValueError(f"{self.imle_configs.weight_edges} not supported")

        if self.imle_configs.marginals_mask or not train:
            sampled_edge_weights = sampled_edge_weights * node_mask

        # B x E x VE
        edge_weight = sampled_edge_weights.permute((1, 2, 3, 4, 0))[real_node_node_mask]
        edge_weight = edge_weight.permute(2, 1, 0).reshape(VE, -1)

        if self.include_original_graph:
            edge_weight = torch.cat([edge_weight, edge_weight.new_ones(VE, dat_batch.num_edges)], dim=1)

        new_graphs = [g.clone() for g in graphs]
        batchsize = len(new_graphs)
        for g in new_graphs:
            g.edge_index = torch.from_numpy(np.vstack(np.triu_indices(g.num_nodes, k=-g.num_nodes))).to(self.device)
        new_graphs = new_graphs * E
        if self.include_original_graph:
            new_graphs += graphs
        new_batch = Batch.from_data_list(new_graphs)
        new_batch.y = dat_batch.y
        new_batch.inter_graph_idx = torch.arange(batchsize).to(self.device).repeat(E + int(self.include_original_graph))

        if train:
            assert VE == 1, "Does not support voting during training"
            new_batch.edge_weight = edge_weight.squeeze()
            return new_batch, output_logits.detach() * real_node_node_mask[..., None], auxloss
        else:
            # we have majority voting batching
            new_batches = []
            for i in range(VE):
                g = new_batch.clone()
                nonzero_idx = edge_weight[i].nonzero().reshape(-1)
                g.edge_index = new_batch.edge_index[:, nonzero_idx]
                g.edge_weight = edge_weight[i, nonzero_idx]
                new_batches.append(g)
            new_batches = Batch.from_data_list(new_batches)
            # batchsize * E
            num_graphs_per_val_ensemble = dat_batch.num_graphs * (E + int(self.include_original_graph))
            inter_graph_idx_rel = torch.arange(VE, device=self.device).repeat_interleave(num_graphs_per_val_ensemble) * dat_batch.num_graphs
            new_batches.inter_graph_idx += inter_graph_idx_rel
            # new_batches.val_ensemble_idx = torch.arange(dat_batch.num_graphs, device=self.device).repeat(VE)
            return new_batches, output_logits.detach() * real_node_node_mask[..., None], auxloss


    def plot(self, new_batch: Batch):
        """Plot graphs and edge weights.        
        Inputs: Batch of graphs from list.
        Outputs: None"""

        # in this case, the batch indices are already modified for inter-subgraph graph pooling,
        # use the original batch instead for compatibility of `to_data_list`
        del new_batch.y  # without repitition, so might be incompatible

        new_batch_plot = new_batch.to_data_list()
        if hasattr(new_batch, 'edge_weight') and new_batch.edge_weight is not None:
            src, dst = new_batch.edge_index
            sections = degree(new_batch.batch[src], dtype=torch.long).tolist()
            weights_split = torch.split(new_batch.edge_weight, split_size_or_sections=sections)
        else:
            weights_split = [None] * len(new_batch_plot)

        n_ensemble = self.sample_configs.ensemble
        total_graphs = len(new_batch_plot)
        unique_graphs = total_graphs // (n_ensemble + int(self.include_original_graph))
        total_plotted_graphs = min(self.plot_args.n_graphs, unique_graphs)

        nrows = round(sqrt(n_ensemble + int(self.include_original_graph)))
        ncols = ceil(sqrt(n_ensemble + int(self.include_original_graph)))
        is_only_one_plot = ncols * nrows == 1

        plots = []
        for graph_id in range(total_plotted_graphs):

            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*7), gridspec_kw={'wspace': 0, 'hspace': 0.05})
            if not is_only_one_plot:
                axs = axs.flatten()
                for ax in axs:
                    ax.set_axis_off()
            else:
                axs.set_axis_off()
            plots.append({'fig': fig, 'axs': axs, 'subgraph_cnt': 0})

        # actually these may not be original graphs, if original graphs not included
        original_graphs_pyg = new_batch_plot[-unique_graphs:]
        original_graphs_pos_dict = []
        for g in original_graphs_pyg:
            original_graphs_pos_np = g.nx_layout.cpu().numpy()
            original_graphs_pos_dict.append({i: pos for i, pos in enumerate(original_graphs_pos_np)})

        if self.include_original_graph:
            original_graphs_networkx = [
                to_networkx(
                    g,
                    to_undirected=True,
                    remove_self_loops=True
                ) for g in original_graphs_pyg
            ]
        else:
            original_graphs_networkx = None

        for i, (g, weights) in enumerate(zip(new_batch_plot, weights_split)):
            graph_id = i % unique_graphs

            if graph_id >= total_plotted_graphs:
                continue

            graph_version = 'original' if i >= (unique_graphs * n_ensemble) else 'rewired'
            if graph_version == 'rewired':
                g_nx = to_networkx(g)
            else:
                assert original_graphs_networkx is not None
                g_nx = original_graphs_networkx[graph_id]

            # Plot g_nx and with edge weights based on "weights". No edge is entry is 0, edge if entry is 1
            if weights is not None:
                edges = g.edge_index[:, torch.where(weights)[0]].T.tolist()
            else:
                edges = g.edge_index.T.tolist()

            node_colors = g.x.detach().cpu().argmax(dim=1).unsqueeze(-1)

            pos = original_graphs_pos_dict[graph_id]

            if not is_only_one_plot:
                ax = plots[graph_id]['axs'][plots[graph_id]['subgraph_cnt']]
                plots[graph_id]['subgraph_cnt'] += 1
            else:
                ax = plots[graph_id]['axs']

            nx.draw_networkx_nodes(g_nx, pos, node_size=200, node_color=node_colors, alpha=0.7, ax=ax)
            nx.draw_networkx_edges(g_nx, pos, edgelist=edges, width=1, edge_color='k', ax=ax)
            nx.draw_networkx_labels(g_nx, pos=pos, labels={i: i for i in range(len(node_colors))}, ax=ax)
            ax.set_title(f'Graph {graph_id}, Epoch {self.epoch}, Version: {graph_version}')

        if hasattr(self.plot_args, 'plot_folder'):
            plot_folder = self.plot_args.plot_folder
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)

        for graph_id, plot in enumerate(plots):
            if hasattr(self.plot_args, 'plot_folder'):
                plot['fig'].savefig(os.path.join(plot_folder, f'e_{self.epoch}_graph_{graph_id}.png'), bbox_inches='tight')

            if self.wandb is not None and self.use_wandb:
                self.wandb.log({f"graph_{graph_id}": self.wandb.Image(plot['fig'])}, step=self.epoch)

            plt.close(plot['fig'])


    def plot_score(self, scores: torch.Tensor):
        scores = scores.cpu().numpy()
        n_ensemble = scores.shape[-1]
        unique_graphs = scores.shape[0]
        total_plotted_graphs = min(self.plot_args.n_graphs, unique_graphs)

        nrows = round(sqrt(n_ensemble))
        ncols = ceil(sqrt(n_ensemble))
        is_only_one_plot = ncols * nrows == 1

        if hasattr(self.plot_args, 'plot_folder'):
            plot_folder = self.plot_args.plot_folder
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)

        for graph_id in range(total_plotted_graphs):
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*7), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
            if not is_only_one_plot:
                axs = axs.flatten()
                for ax in axs:
                    ax.set_axis_off()
            else:
                axs.set_axis_off()

            atten_scores = scores[graph_id, ...]
            for e in range(n_ensemble):
                sns.heatmap(atten_scores[..., e], ax=axs[e] if not is_only_one_plot else axs)

            if hasattr(self.plot_args, 'plot_folder'):
                fig.savefig(os.path.join(plot_folder,
                                        f'scores_epoch_{self.epoch}_graph_{graph_id}.png'),
                            bbox_inches='tight')

            if self.wandb is not None and self.use_wandb:
                self.wandb.log({f"scores_{graph_id}": self.wandb.Image(fig)}, step=self.epoch)

            plt.close(fig)

    def train(self,
              dataloader: AttributedDataLoader,
              emb_model: Emb_model,
              model: Train_model,
              optimizer_embd: Optional[Optimizer],
              optimizer: Optimizer):

        if emb_model is not None:
            emb_model.train()
            optimizer_embd.zero_grad()

        model.train()
        train_losses = torch.tensor(0., device=self.device)
        preds = []
        labels = []
        num_graphs = 0

        for batch_id, data in enumerate(dataloader.loader):
            optimizer.zero_grad()
            data = self.check_datatype(data)
            data, scores, auxloss = self.construct_duplicate_data(data, emb_model)

            pred = model(data)

            is_labeled = data.y == data.y
            loss = self.criterion(pred[is_labeled], data.y[is_labeled])
            train_losses += loss.detach() * data.num_graphs

            if auxloss is not None:
                loss = loss + auxloss

            loss.backward()
            optimizer.step()
            if optimizer_embd is not None:
                if (batch_id % self.micro_batch_embd == self.micro_batch_embd - 1) or (batch_id >= len(dataloader) - 1):
                    emb_model = scale_grad(emb_model, (batch_id % self.micro_batch_embd) + 1)
                    # torch.nn.utils.clip_grad_value_(emb_model.parameters(), clip_value=1.0)
                    torch.nn.utils.clip_grad_norm_(emb_model.parameters(), max_norm=1.0, error_if_nonfinite=False)
                    optimizer_embd.step()
                    optimizer_embd.zero_grad()

            num_graphs += data.num_graphs
            preds.append(pred)
            labels.append(data.y)

            if self.plot_args is not None:
                if batch_id == self.plot_args.batch_id and self.epoch % self.plot_args.plot_every == 0:
                    self.plot(data)
                    if scores is not None:
                        self.plot_score(scores)

        train_loss = train_losses.item() / num_graphs
        self.best_train_loss = min(self.best_train_loss, train_loss)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        train_metric = get_eval(self.task_type, labels, preds)

        return train_loss, train_metric

    @torch.no_grad()
    def inference(self,
                  dataloader: AttributedDataLoader,
                  emb_model: Emb_model,
                  model: Train_model,
                  scheduler_embd: Optional[Scheduler] = None,
                  scheduler: Optional[Scheduler] = None,
                  train_loss: float = 0.,
                  train_metric: float = 0.,
                  test: bool = False):
        if emb_model is not None:
            emb_model.eval()

        model.eval()
        preds = []
        labels = []

        preds_ensemble = []
        labels_ensemble = []

        preds_uncertainty = []

        for data in dataloader.loader:
            data = self.check_datatype(data)
            if isinstance(data, Data):
                num_graphs = data.num_graphs
            else:
                num_graphs = len(data[1])
            data, _, _ = self.construct_duplicate_data(data, emb_model)

            pred = model(data)
            label = data.y
            label_ensemble = label[:num_graphs]

            pred_ensemble = pred.reshape(*(-1, num_graphs) + pred.shape[1:]).transpose(0, 1)

            # For classification we should use something like the entropy of the mean prediction
            # i.e. we have (B_SZ, N_ENS, N_CLASSES)->(B_SZ, 1, N_CLASSES)->compute entropy over N_CLASSES
            pred_uncertainty = pred_ensemble.cpu().numpy().std(1)
            pred_ensemble = pred_ensemble.mean(dim=1)

            dataset_std = 1 if dataloader.std is None else dataloader.std
            preds.append(pred * dataset_std)
            preds_ensemble.append(pred_ensemble * dataset_std)
            labels.append(label * dataset_std)
            labels_ensemble.append(label_ensemble * dataset_std)
            preds_uncertainty.append(pred_uncertainty)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        preds_ensemble = torch.cat(preds_ensemble, dim=0)
        labels_ensemble = torch.cat(labels_ensemble, dim=0)
        preds_uncertainty = np.concatenate(preds_uncertainty, axis=0)

        is_labeled = labels == labels
        is_labeled_ens = labels_ensemble == labels_ensemble
        val_loss = self.criterion(preds[is_labeled], labels[is_labeled]).item()
        val_loss_ensemble = self.criterion(preds_ensemble[is_labeled_ens], labels_ensemble[is_labeled_ens]).item()

        val_metric = get_eval(self.task_type, labels, preds)
        val_metric_ensemble = get_eval(self.task_type, labels_ensemble, preds_ensemble)

        early_stop = False
        if not test:
            if self.wandb is not None and self.use_wandb:
                self.wandb.log({"train_loss": train_loss,
                                "train_metric": train_metric,
                                "val_loss": val_loss,
                                "val_metric": val_metric,
                                "val_loss_ensemble": val_loss_ensemble,
                                "val_metric_ensemble": val_metric_ensemble,
                                "down_lr": scheduler.get_last_lr()[-1],
                                "up_lr": scheduler_embd.get_last_lr()[-1] if emb_model is not None else 0.,
                                "val_preds_uncertainty": self.wandb.Histogram(preds_uncertainty)})

            self.best_val_loss = min(self.best_val_loss, val_loss)

            if isinstance(scheduler, MyPlateau):
                if scheduler.lr_target == 'train_metric':
                    scheduler.step(train_metric)
                elif scheduler.lr_target == 'train_loss':
                    scheduler.step(train_loss)
                elif scheduler.lr_target == 'val_metric':
                    scheduler.step(val_metric)
                elif scheduler.lr_target == 'val_loss':
                    scheduler.step(val_loss)
            else:
                scheduler.step()
            if scheduler_embd is not None:
                if isinstance(scheduler_embd, MyPlateau):
                    raise NotImplementedError("Need to specify max or min plateau")
                else:
                    scheduler_embd.step()

            train_is_better, self.best_train_metric = self.metric_comparator(train_metric, self.best_train_metric)
            val_is_better, self.best_val_metric = self.metric_comparator(val_metric, self.best_val_metric)

            if self.patience_target == 'val_metric':
                is_better = val_is_better
            elif self.patience_target == 'train_metric':
                is_better = train_is_better
            else:
                raise NotImplementedError

            if is_better:
                self.patience = 0
            else:
                self.patience += 1
                if self.patience > self.max_patience:
                    early_stop = True

        return val_loss, val_metric, val_loss_ensemble, val_metric_ensemble, early_stop

    def clear_stats(self):
        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.best_train_loss = 1e5
        self.best_train_metric = None
        self.patience = 0
