# credits to OGB team
# https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/gnn.py
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_scatter import scatter

from .ogb_mol_conv import GNN_node, GNN_node_Virtualnode, GNN_node_order


# TODO: follow NGNN model
# TODO: split inner and outer GNN

class OGBGNN(torch.nn.Module):

    def __init__(self,
                 num_tasks,
                 num_layer=5,
                 emb_dim=300,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 subgraph2node_aggr='add',
                 graph_pooling="mean"):

        super(OGBGNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.subgraph2node_aggr = subgraph2node_aggr

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                                 gnn_type=gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                     gnn_type=gnn_type)

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                            torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if self.subgraph2node_aggr in ['center', 'add']:
            self.inner_pool = global_add_pool
        elif self.subgraph2node_aggr is None:
            pass
        else:
            raise NotImplementedError

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, data):
        h_node = self.gnn_node(data)

        if hasattr(data, 'node_mask') and hasattr(data, 'nodes2graph'):
            if data.node_mask.dtype == torch.float:
                h_node = h_node * data.node_mask[:, None]
            else:
                h_node = h_node[data.node_mask]
            h_node = self.inner_pool(h_node, data.subgraphs2nodes)
            h_graph = self.pool(h_node, data.nodes2graph)
        else:
            h_graph = self.pool(h_node, data.batch)

        return self.graph_pred_linear(h_graph)

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        if isinstance(self.pool, (GlobalAttention, Set2Set)):
            self.pool.reset_parameters()
        self.graph_pred_linear.reset_parameters()
