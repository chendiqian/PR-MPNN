import os
from math import ceil, sqrt

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import torch
from ml_collections import ConfigDict
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx, degree


def plot_rewired_graphs(new_batch: Batch,
                        train: bool,
                        epoch,
                        batch_id,
                        wandb,
                        use_wandb,
                        plot_args,
                        ensemble,
                        num_train_ensemble,
                        num_val_ensemble,
                        include_original_graph):

    if batch_id != plot_args.batch_id or epoch % plot_args.plot_every != 0:
        return

    if hasattr(plot_args, 'plot_folder'):
        plot_folder = plot_args.plot_folder
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

    # in this case, the batch indices are already modified for inter-subgraph graph pooling,
    # use the original batch instead for compatibility of `to_data_list`
    del new_batch.y  # without repitition, so might be incompatible

    new_batch_plot = new_batch.to_data_list()
    if hasattr(new_batch, 'edge_weight') and new_batch.edge_weight is not None:
        src, dst = new_batch.edge_index
        sections = degree(new_batch.batch[src], dtype=torch.long).tolist()
        weights_split = torch.split(new_batch.edge_weight,
                                    split_size_or_sections=sections)
    else:
        weights_split = [None] * len(new_batch_plot)

    n_ensemble = ensemble * (num_train_ensemble if train else num_val_ensemble)
    unique_graphs = len(new_batch_plot) // n_ensemble // (
        2 if include_original_graph else 1)
    total_plotted_graphs = min(plot_args.n_graphs, unique_graphs)

    nrows = round(sqrt(n_ensemble + int(include_original_graph)))
    ncols = ceil(sqrt(n_ensemble + int(include_original_graph)))
    is_only_one_plot = ncols * nrows == 1

    for graph_id in range(total_plotted_graphs):
        g_nx = to_networkx(new_batch_plot[graph_id])

        original_graphs_pos_np = new_batch_plot[graph_id].nx_layout.cpu().numpy()
        original_graphs_pos_dict = {i: pos for i, pos in
                                    enumerate(original_graphs_pos_np)}

        if 'dataset' in plot_args and plot_args.dataset == 'leafcolor':
            node_colors = new_batch_plot[graph_id].x[:, 1].detach().cpu().unsqueeze(-1)
        elif 'dataset' in plot_args and plot_args.dataset == 'trees':
            idx_label = torch.where(new_batch_plot[graph_id].x[:, 1] == new_batch_plot[graph_id].x[0][0])[0].item()
            node_colors = torch.tensor([0 for _ in range(new_batch_plot[graph_id].x.shape[0])])
            node_colors[idx_label] = 1
            node_colors = node_colors.unsqueeze(-1)
        else:
            node_colors = new_batch_plot[graph_id].x.detach().cpu().argmax(
                dim=1).unsqueeze(-1)

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 7, nrows * 7),
                                gridspec_kw={'wspace': 0, 'hspace': 0.05})
        if not is_only_one_plot:
            axs = axs.flatten()
            for ax in axs:
                ax.set_axis_off()
        else:
            axs.set_axis_off()

        for variant_id in range(n_ensemble + int(include_original_graph)):
            graph_version = 'original' if variant_id == n_ensemble else 'rewired'
            lst_id = variant_id * unique_graphs + graph_id
            g = new_batch_plot[lst_id]

            weights = weights_split[lst_id]
            if weights is not None:
                edges = g.edge_index[:, torch.where(weights)[0]].T.tolist()
            else:
                edges = g.edge_index.T.tolist()

            if not is_only_one_plot:
                ax = axs[variant_id]
            else:
                ax = axs

            nx.draw_networkx_nodes(g_nx, original_graphs_pos_dict, node_size=200,
                                   node_color=node_colors, alpha=0.7, ax=ax)
            nx.draw_networkx_edges(g_nx, original_graphs_pos_dict, edgelist=edges,
                                   width=1, edge_color='k', ax=ax)
            nx.draw_networkx_labels(g_nx, pos=original_graphs_pos_dict,
                                    labels={i: i for i in range(len(node_colors))}, ax=ax)
            ax.set_title(f'Graph {graph_id}, Epoch {epoch}, Version: {graph_version}')

        if hasattr(plot_args, 'plot_folder'):
            fig.savefig(os.path.join(plot_folder, f'e_{epoch}_graph_{graph_id}.png'),
                        bbox_inches='tight')

        if wandb is not None and use_wandb:
            wandb.log({f"graph_{graph_id}": wandb.Image(fig)}, step=epoch)

        plt.close(fig)


def plot_score(scores: torch.Tensor,
               epoch,
               batch_id,
               plot_args: ConfigDict,
               wandb,
               use_wandb: bool):
    if batch_id != plot_args.batch_id or epoch % plot_args.plot_every != 0:
        return

    scores = scores.cpu().numpy()
    n_ensemble = scores.shape[-1]
    unique_graphs = scores.shape[0]
    total_plotted_graphs = min(plot_args.n_graphs, unique_graphs)

    nrows = round(sqrt(n_ensemble))
    ncols = ceil(sqrt(n_ensemble))
    is_only_one_plot = ncols * nrows == 1

    if hasattr(plot_args, 'plot_folder'):
        plot_folder = plot_args.plot_folder
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

    for graph_id in range(total_plotted_graphs):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols *7, nrows *7), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
        if not is_only_one_plot:
            axs = axs.flatten()
            for ax in axs:
                ax.set_axis_off()
        else:
            axs.set_axis_off()

        atten_scores = scores[graph_id, ...]
        for e in range(n_ensemble):
            sns.heatmap(atten_scores[..., e], ax=axs[e] if not is_only_one_plot else axs)

        if hasattr(plot_args, 'plot_folder'):
            fig.savefig(os.path.join(plot_folder, f'scores_epoch_{epoch}_graph_{graph_id}.png'),
                        bbox_inches='tight')

        if wandb is not None and use_wandb:
            wandb.log({f"scores_{graph_id}": wandb.Image(fig)}, step=epoch)

        plt.close(fig)
