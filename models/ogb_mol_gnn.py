# credits to OGB team
# https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/gnn.py
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_scatter import scatter
from .encoder import AtomEncoder
from .ogb_mol_conv import GNN_node, GNN_node_Placeholder


class OGBGNN(torch.nn.Module):

    def __init__(self,
                 num_tasks,
                 extra_dim=None,
                 num_layer=5,
                 emb_dim=300,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean"):
        """

        :param num_tasks:
        :param extra_dim: a list of 2, indicating the embedding dims of x in the original graph and and
        embeddings of nested subgraphs
        :param num_layer:
        :param emb_dim:
        :param gnn_type:
        :param virtual_node:
        :param residual:
        :param drop_ratio:
        :param JK:
        :param graph_pooling:
        """

        super(OGBGNN, self).__init__()

        if extra_dim is None:
            extra_dim = [300, 300]
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        # map data.x to dim0
        self.atom_encoder = AtomEncoder(extra_dim[0]) if extra_dim[0] > 0 else None
        # map the extra feature to dim1
        self.extra_emb_layer = torch.nn.Linear(emb_dim, extra_dim[1]) if extra_dim[1] > 0 else None
        # merge 2 features
        self.merge_layer = torch.nn.Linear(sum(extra_dim), emb_dim) if min(extra_dim) > 0 else None

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
        if self.atom_encoder is not None and self.extra_emb_layer is not None:
            x = self.atom_encoder(data.x)
            extra = self.extra_emb_layer(intermediate_node_emb)
            input_data = torch.relu(
                self.merge_layer(
                    torch.cat([x, extra], dim=1)
                )
            )
        elif self.atom_encoder is not None and self.extra_emb_layer is None:
            input_data = self.atom_encoder(data.x)
        elif self.atom_encoder is None and self.extra_emb_layer is not None:
            input_data = self.extra_emb_layer(intermediate_node_emb)
        else:
            raise ValueError

        data.x = input_data
        h_node = self.gnn_node(data)
        h_graph = self.pool(h_node, data.batch)
        return self.graph_pred_linear(h_graph)

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        if isinstance(self.pool, (GlobalAttention, Set2Set)):
            self.pool.reset_parameters()
        self.graph_pred_linear.reset_parameters()
        if self.atom_encoder is not None:
            self.atom_encoder.reset_parameters()
        if self.extra_emb_layer is not None:
            self.extra_emb_layer.reset_parameters()
        if self.merge_layer is not None:
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
            self.inner_pool = lambda x, subgraphs2nodes, *args: global_add_pool(x, subgraphs2nodes)
        elif self.graph_pooling == 'mean':
            self.inner_pool = \
                lambda x, subgraphs2nodes, node_mask: global_add_pool(x, subgraphs2nodes) / \
                                                      scatter(node_mask.detach(),
                                                              subgraphs2nodes, dim=0, reduce="sum")[:, None]
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
            h_node = self.inner_pool(h_node, data.subgraphs2nodes, data.node_mask)

        return h_node

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
