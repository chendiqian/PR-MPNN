import os
import pickle
from collections import defaultdict
from math import ceil, sqrt
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as pyg_utils
from ml_collections import ConfigDict
from torch_geometric.data import Batch, Data

from data.data_utils import AttributedDataLoader, IsBetter, scale_grad
from data.metrics import eval_acc, eval_rmse, eval_rocauc
from imle.noise import GumbelDistribution
from imle.target import TargetDistribution
from imle.wrapper import imle
from training.aux_loss import get_kl_aux_loss, get_norm_aux_loss
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
                 criterion: Loss,
                 device: Union[str, torch.device],
                 imle_configs: ConfigDict,
                 sample_configs: ConfigDict,
                 aux_type: str,
                 auxloss: float = 0.,
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

        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.patience = 0
        self.max_patience = max_patience
        self.aux_type = aux_type
        self.auxloss = auxloss

        self.wandb = wandb
        self.use_wandb = use_wandb
        self.plot_args = plot_args

        self.curves = defaultdict(list)

        self.epoch = None

        self.sample_policy = sample_configs.sample_policy
        self.include_original_graph = sample_configs.include_original_graph
        if imle_configs is not None:
            # upstream may have micro batching
            self.micro_batch_embd = imle_configs.micro_batch_embd

            if imle_configs.sampler == 'imle':
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

            elif imle_configs.sampler == 'gumbel':
                assert sample_configs.sample_policy == 'global_topk'
                gumbel_sampler = GumbelSampler(sample_configs.sample_k, tau=imle_configs.tau)

                self.train_forward = gumbel_sampler
                self.val_forward = gumbel_sampler.validation

            elif imle_configs.sampler == 'simple':
                assert sample_configs.sample_policy == 'global_topk'
                simple_sampler = EdgeSIMPLEBatched(sample_configs.sample_k, device)
                self.train_forward = simple_sampler
                self.val_forward = simple_sampler.validation
            else:
                raise ValueError

        if sample_configs.sample_policy is None or imle_configs is None:
            self.construct_duplicate_data = lambda x, *args: (x, None)
        elif imle_configs is not None:
            self.construct_duplicate_data = self.emb_model_graph_level_pred


    def emb_model_graph_level_pred(self,
                                   dat_batch: Union[Batch, Data],
                                   emb_model: Emb_model,):

        train = emb_model.training
        logits, real_node_node_mask = emb_model(dat_batch)

        padding_bias = (~real_node_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
        logits = logits - padding_bias
        node_mask, _ = self.train_forward(logits) if train else self.val_forward(logits)

        if self.auxloss > 0 and train:
            if self.aux_type == 'entropy':
                auxloss = get_kl_aux_loss(node_mask * real_node_node_mask[..., None].to(torch.float), self.auxloss)
            elif self.aux_type == 'norm':
                auxloss = get_norm_aux_loss(logits, self.auxloss, real_node_node_mask)
            else:
                raise ValueError
        else:
            auxloss = None

        graphs = Batch.to_data_list(dat_batch)
        batchsize = len(graphs)
        for g in graphs:
            g.edge_index = torch.from_numpy(np.vstack(np.triu_indices(g.num_nodes, k=-g.num_nodes))).to(self.device)
        graphs = graphs * logits.shape[-1]
        if self.include_original_graph:
            original_graphs = Batch.to_data_list(dat_batch)
            graphs += original_graphs

        edge_weight = node_mask[real_node_node_mask].T.reshape(-1)
        if self.include_original_graph:
            edge_weight = torch.cat([edge_weight, edge_weight.new_ones(dat_batch.num_edges)], dim=0)
        new_batch = Batch.from_data_list(graphs)
        if train:
            new_batch.edge_weight = edge_weight
        else:
            new_batch.edge_index = new_batch.edge_index[:, edge_weight.to(torch.bool)]
            new_batch.edge_weight = None

        new_batch.y = dat_batch.y
        new_batch.inter_graph_idx = torch.arange(batchsize).to(self.device).repeat(logits.shape[-1] + int(self.include_original_graph))
        return new_batch, auxloss

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
            sections = pyg_utils.degree(new_batch.batch[src], dtype=torch.long).tolist()
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
                pyg_utils.convert.to_networkx(
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
                g_nx = pyg_utils.convert.to_networkx(g)
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

        plot_folder = self.plot_args.plot_folder
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        for graph_id, plot in enumerate(plots):
            plot['fig'].savefig(os.path.join(plot_folder, f'e_{self.epoch}_graph_{graph_id}.png'), bbox_inches='tight')

            if self.wandb is not None and self.use_wandb:
                self.wandb.log({f"graph_{graph_id}": self.wandb.Image(plot['fig'])}, step=self.epoch)

            plt.close(plot['fig'])


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
        if self.task_type != 'regression':
            preds = []
            labels = []
        else:
            preds, labels = None, None
        num_graphs = 0

        for batch_id, data in enumerate(dataloader.loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            data, auxloss = self.construct_duplicate_data(data, emb_model)

            pred = model(data)

            is_labeled = data.y == data.y
            loss = self.criterion(pred[is_labeled], data.y[is_labeled].to(torch.float))
            train_losses += loss.detach() * data.num_graphs

            if auxloss is not None:
                loss = loss + auxloss

            loss.backward()
            optimizer.step()
            if optimizer_embd is not None:
                if (batch_id % self.micro_batch_embd == self.micro_batch_embd - 1) or (batch_id >= len(dataloader) - 1):
                    emb_model = scale_grad(emb_model, (batch_id % self.micro_batch_embd) + 1)
                    torch.nn.utils.clip_grad_value_(emb_model.parameters(), clip_value=1.0)
                    optimizer_embd.step()
                    optimizer_embd.zero_grad()

            num_graphs += data.num_graphs
            if isinstance(preds, list):
                preds.append(pred)
                labels.append(data.y)

            if self.plot_args is not None:
                if batch_id == self.plot_args.batch_id and self.epoch % self.plot_args.plot_every == 0:
                    self.plot(data)

        train_loss = train_losses.item() / num_graphs
        self.curves['train_loss'].append(train_loss)

        if isinstance(preds, list):
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            if self.task_type == 'rocauc':
                train_metric = eval_rocauc(labels, preds)
            elif self.task_type == 'rmse':
                train_metric = eval_rmse(labels, preds)
            elif self.task_type == 'acc':
                if preds.shape[1] == 1:
                    preds = (preds > 0.).to(torch.int)
                else:
                    preds = torch.argmax(preds, dim=1)
                train_metric = eval_acc(labels, preds)
            else:
                raise NotImplementedError
        else:
            train_metric = train_loss
        self.curves['train_metric'].append(train_metric)

        return train_loss, train_metric

    @torch.no_grad()
    def inference(self,
                  dataloader: AttributedDataLoader,
                  emb_model: Emb_model,
                  model: Train_model,
                  scheduler_embd: Optional[Scheduler] = None,
                  scheduler: Optional[Scheduler] = None,
                  test: bool = False):
        if emb_model is not None:
            emb_model.eval()

        model.eval()
        preds = []
        labels = []

        for data in dataloader.loader:
            data = data.to(self.device)
            data, _ = self.construct_duplicate_data(data, emb_model)

            pred = model(data)

            if dataloader.std is not None:
                preds.append(pred * dataloader.std)
                labels.append(data.y.to(torch.float) * dataloader.std)
            else:
                preds.append(pred)
                labels.append(data.y.to(torch.float))

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        is_labeled = labels == labels
        val_loss = self.criterion(preds[is_labeled], labels[is_labeled]).item()
        if self.task_type == 'rocauc':
            val_metric = eval_rocauc(labels, preds)
        elif self.task_type == 'regression':  # not specified regression loss type
            val_metric = val_loss
        elif self.task_type == 'rmse':
            val_metric = eval_rmse(labels, preds)
        elif self.task_type == 'acc':
            if preds.shape[1] == 1:
                preds = (preds > 0.).to(torch.int)
            else:
                preds = torch.argmax(preds, dim=1)
            val_metric = eval_acc(labels, preds)
        else:
            raise NotImplementedError

        early_stop = False
        if not test:
            self.curves['val_metric'].append(val_metric)
            self.curves['val_loss'].append(val_loss)

            self.best_val_loss = min(self.best_val_loss, val_loss)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                raise NotImplementedError("Need to specify max or min plateau")
            else:
                scheduler.step()
            if scheduler_embd is not None:
                if isinstance(scheduler_embd, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    raise NotImplementedError("Need to specify max or min plateau")
                else:
                    scheduler_embd.step()

            if self.metric_comparator(val_metric, self.best_val_metric):
                self.best_val_metric = val_metric
                self.patience = 0
            else:
                self.patience += 1
                if self.patience > self.max_patience:
                    early_stop = True
        return val_loss, val_metric, early_stop

    def save_curve(self, path):
        pickle.dump(self.curves, open(os.path.join(path, 'curves.pkl'), "wb"))

    def clear_stats(self):
        self.curves = defaultdict(list)
        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.patience = 0
