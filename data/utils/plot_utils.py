import math
import os
from math import ceil, sqrt

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from ml_collections import ConfigDict
from torch_geometric.utils import to_networkx, to_undirected

from data.utils.datatype_utils import DuoDataStructure
from data.utils.neighbor_utils import get_khop_neighbors


def plot_rewired_graphs(new_batch: DuoDataStructure,
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

    assert isinstance(new_batch, DuoDataStructure), f"unsupported dtype {type(new_batch)}"

    if hasattr(plot_args, 'plot_folder'):
        plot_folder = plot_args.plot_folder
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

    if include_original_graph:
        assert new_batch.org is not None
        original_graph = new_batch.org.to_data_list()

    candidates = new_batch.candidates
    per_layer_sampled = isinstance(candidates[0].edge_weight, (list, tuple))
    if per_layer_sampled:
        L = len(candidates[0].edge_weight)

    plot_per_graph = ensemble * (num_train_ensemble if train else num_val_ensemble) * len(candidates) * (L if per_layer_sampled else 1) + int(include_original_graph)
    n_ensemble = ensemble * (num_train_ensemble if train else num_val_ensemble)
    unique_graphs = new_batch.num_graphs // n_ensemble
    num_graphs_2b_plot = min(unique_graphs, plot_args.n_graphs)
    graphs = [[] for _ in range(len(candidates))]
    for i_c, c in enumerate(candidates):
        if per_layer_sampled:
            for i in range(L):
                graph = c.clone()
                if isinstance(graph.edge_index, (list, tuple)):
                    graph.edge_index = graph.edge_index[i]
                if isinstance(graph.edge_attr, (list, tuple)):
                    graph.edge_attr = graph.edge_attr[i]
                if isinstance(graph.edge_weight, (list, tuple)):
                    graph.edge_weight = graph.edge_weight[i]
                if isinstance(graph._slice_dict['edge_index'], (list, tuple)):
                    graph._slice_dict['edge_index'] = graph._slice_dict['edge_index'][i]
                if isinstance(graph._slice_dict['edge_attr'], (list, tuple)):
                    graph._slice_dict['edge_attr'] = graph._slice_dict['edge_attr'][i]
                if isinstance(graph._slice_dict['edge_weight'], (list, tuple)):
                    graph._slice_dict['edge_weight'] = graph._slice_dict['edge_weight'][i]

                graphs[i_c].append(graph.to_data_list())
        else:
            graphs[i_c].append(c.to_data_list())


    # in this case, the batch indices are already modified for inter-subgraph graph pooling,
    # use the original batch instead for compatibility of `to_data_list`
    nrows = round(sqrt(plot_per_graph))
    ncols = ceil(sqrt(plot_per_graph))
    is_only_one_plot = ncols * nrows == 1

    for graph_id in range(num_graphs_2b_plot):
        g_nx = to_networkx(graphs[0][0][graph_id])

        original_graphs_pos_np = graphs[0][0][graph_id].nx_layout.cpu().numpy()
        original_graphs_pos_dict = {i: pos for i, pos in enumerate(original_graphs_pos_np)}

        if 'dataset' in plot_args and plot_args.dataset == 'leafcolor':
            node_colors = graphs[0][0][graph_id].x[:, 1].detach().cpu().unsqueeze(-1)
        elif 'dataset' in plot_args and plot_args.dataset == 'trees':
            idx_label = torch.where(graphs[0][0][graph_id].x[:, 1] == graphs[0][0][graph_id].x[0][0])[0].item()
            node_colors = torch.tensor([0 for _ in range(graphs[0][0][graph_id].x.shape[0])])
            node_colors[idx_label] = 1
            node_colors = node_colors.unsqueeze(-1)
        else:
            node_colors = graphs[0][0][graph_id].x.detach().cpu().argmax(dim=1).unsqueeze(-1)

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 7, nrows * 7),
                                gridspec_kw={'wspace': 0, 'hspace': 0.05})
        if not is_only_one_plot:
            axs = axs.flatten()
            for ax in axs:
                ax.set_axis_off()
        else:
            axs.set_axis_off()

        ax_idx = 0
        if include_original_graph:
            ax = axs[0]
            g = original_graph[graph_id]
            edges = g.edge_index.T.tolist()
            nx.draw_networkx_nodes(g_nx, original_graphs_pos_dict, node_size=200, node_color=node_colors, alpha=0.7, ax=ax)
            nx.draw_networkx_edges(g_nx, original_graphs_pos_dict, edgelist=edges, width=1, edge_color='k', ax=ax)
            nx.draw_networkx_labels(g_nx, pos=original_graphs_pos_dict, labels={i: i for i in range(len(node_colors))}, ax=ax)
            ax.set_title(f'Graph {graph_id}, Epoch {epoch}, Version: original')
            ax_idx = 1

        for i_c, c in enumerate(graphs):
            for i_l, layer in enumerate(c):
                for i_g in range(n_ensemble):
                    g = layer[i_g * unique_graphs + graph_id]
                    if g.edge_weight is not None:
                        edges = g.edge_index[:, torch.where(g.edge_weight)[0]].T.tolist()
                    else:
                        edges = g.edge_index.T.tolist()

                    if not is_only_one_plot:
                        ax = axs[ax_idx]
                    else:
                        ax = axs

                    nx.draw_networkx_nodes(g_nx, original_graphs_pos_dict, node_size=200, node_color=node_colors, alpha=0.7, ax=ax)
                    nx.draw_networkx_edges(g_nx, original_graphs_pos_dict, edgelist=edges, width=1, edge_color='k', ax=ax)
                    nx.draw_networkx_labels(g_nx, pos=original_graphs_pos_dict, labels={i: i for i in range(len(node_colors))}, ax=ax)
                    ax.set_title(f'Graph {graph_id}, Epoch {epoch}, candidate: {i_c}, layer: {i_l}, ensemble: {i_g}')
                    ax_idx += 1

        if hasattr(plot_args, 'plot_folder'):
            fig.savefig(os.path.join(plot_folder, f'e_{epoch}_graph_{graph_id}.png'), bbox_inches='tight')

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


def circular_tree_layout(graph: nx.DiGraph):
    edge_index = torch.tensor(list(graph.edges)).T
    edge_index = to_undirected(edge_index, num_nodes=len(graph.nodes))
    edge_index = edge_index[:, edge_index[0] < edge_index[1]].cpu().numpy()

    khop_neighbors, _ = get_khop_neighbors(0, edge_index, 10000)  # basically all neighbors
    layout = dict()

    def generate_dots(num_dots, r):
        angle = math.pi / num_dots
        dots = []
        for i in range(num_dots):
            dots.append([math.cos(angle) * r, math.sin(angle) * r])
            angle += 2 * math.pi / num_dots
        return dots

    for i, neighbors in enumerate(khop_neighbors):
        pos = generate_dots(len(neighbors), i * 0.1 * 1.1 ** i)
        for j, n in enumerate(neighbors):
            layout[n] = np.array(pos[j], dtype=np.float64)

    return layout
