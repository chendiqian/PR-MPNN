# credits to OGB team
# https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/gnn.py
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from .encoder import AtomEncoder
from .ogb_mol_conv import GNN_node, GNN_node_Placeholder


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
                 graph_pooling="mean"):

        super(OGBGNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        self.atom_encoder = AtomEncoder(emb_dim)
        self.merge_layer = torch.nn.Linear(2 * emb_dim, emb_dim)

        # GNN to generate node embeddings
        if num_layer > 0:
            if virtual_node:
                raise NotImplementedError
            else:
                self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                         gnn_type=gnn_type, atom_embed=False)
        else:
            self.gnn_node = GNN_node_Placeholder()

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

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, data, intermediate_node_emb):
        x = self.atom_encoder(data.x)
        x = torch.cat([x, intermediate_node_emb], dim=1)
        data.x = torch.relu(self.merge_layer(x))

        h_node = self.gnn_node(data)
        h_graph = self.pool(h_node, data.batch)
        return self.graph_pred_linear(h_graph)

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        if isinstance(self.pool, (GlobalAttention, Set2Set)):
            self.pool.reset_parameters()
        self.graph_pred_linear.reset_parameters()
        self.atom_encoder.reset_parameters()
        self.merge_layer.reset_parameters()


class OGBGNN_inner(torch.nn.Module):

    def __init__(self,
                 num_layer=5,
                 emb_dim=300,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 subgraph2node_aggr='add'):

        super(OGBGNN_inner, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = subgraph2node_aggr

        # GNN to generate node embeddings
        if virtual_node:
            raise NotImplementedError
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                     gnn_type=gnn_type)

        if self.graph_pooling in ['center', 'add']:
            self.inner_pool = global_add_pool
        elif self.graph_pooling is None:
            self.inner_pool = None
        else:
            raise NotImplementedError

    def forward(self, data):
        h_node = self.gnn_node(data)

        if self.inner_pool is not None:
            assert hasattr(data, 'node_mask') and hasattr(data, 'subgraphs2nodes')
            if data.node_mask.dtype == torch.float:
                h_node = h_node * data.node_mask[:, None]
            else:
                h_node = h_node[data.node_mask]
            h_node = self.inner_pool(h_node, data.subgraphs2nodes)

        return h_node

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
