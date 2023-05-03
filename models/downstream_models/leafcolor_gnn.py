import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from models.my_convs import GINConv
from models.nn_modules import BiEmbedding_cat


def get_layer(gnn_type, in_dim, out_dim):
    if gnn_type == 'gin':
        return GINConv(out_dim,
                       nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                                     nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()))
    else:
        raise NotImplementedError


class LeafColorGraphModel(torch.nn.Module):
    def __init__(self, gnn_type, num_layers, tree_depth, n_leaf_labels, h_dim, out_dim, last_layer_fully_adjacent,
                 unroll, layer_norm, use_activation, use_residual):
        super(LeafColorGraphModel, self).__init__()
        self.unroll = unroll
        self.last_layer_fully_adjacent = last_layer_fully_adjacent
        self.use_layer_norm = layer_norm
        self.use_activation = use_activation
        self.use_residual = use_residual
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_layers = num_layers

        self.tree_depth = tree_depth
        self.n_leaf_labels = n_leaf_labels

        total_nodes = 2 ** (tree_depth + 1) - 1

        assert h_dim % 2 == 0, 'h_dim must be even'

        self.embedding = BiEmbedding_cat(total_nodes, n_leaf_labels, int(h_dim/2))
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        if unroll:
            self.layers.append(get_layer(
                gnn_type,
                in_dim=h_dim,
                out_dim=h_dim))
        else:
            for i in range(num_layers):
                self.layers.append(get_layer(
                    gnn_type,
                    in_dim=h_dim,
                    out_dim=h_dim))
        if self.use_layer_norm:
            for i in range(num_layers):
                self.layer_norms.append(nn.LayerNorm(h_dim))

        self.out_dim = out_dim
        self.out_layer = nn.Linear(in_features=h_dim, out_features=out_dim, bias=False)

    def forward(self, data):
        x, edge_index, batch, roots = data.x, data.edge_index, data.batch, data.root_mask
        edge_weight = data.edge_weight

        x = self.embedding(x)

        for i in range(self.num_layers):
            if self.unroll:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            new_x = x
            if self.last_layer_fully_adjacent and i == self.num_layers - 1:
                root_indices = torch.nonzero(roots, as_tuple=False).squeeze(-1)
                target_roots = root_indices.index_select(dim=0, index=batch)
                source_nodes = torch.arange(0, data.num_nodes).to(self.device)
                edges = torch.stack([source_nodes, target_roots], dim=0)

            else:
                edges = edge_index
            new_x = layer(new_x, edges, edge_weight)
            if self.use_activation:
                new_x = F.relu(new_x)
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        root_nodes = x[roots]
        if hasattr(data, 'inter_graph_idx'):
            root_nodes = global_mean_pool(root_nodes, data.inter_graph_idx)
        logits = self.out_layer(root_nodes)
        return logits
