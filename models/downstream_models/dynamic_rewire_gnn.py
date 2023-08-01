from typing import Union

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_batch

from models.my_convs import GINEConv
from models.nn_modules import MLP, TransformerLayer, AttentionLayer
from models.nn_utils import residual


class DynamicRewireGNN(torch.nn.Module):
    def __init__(self,
                 sampler,
                 make_intermediate_gnn,
                 encoder,
                 edge_encoder,
                 hid_size,
                 gnn_type,
                 gnn_layer,
                 sample_mlp_layer,
                 num_classes,
                 directed_sampling,
                 residual,
                 dropout,
                 ensemble,
                 use_bn,
                 mlp_layers_intragraph,
                 graph_pooling,
                 sample_alpha=1):
        super(DynamicRewireGNN, self).__init__()

        if isinstance(sample_mlp_layer, int):
            self.use_bilinear = False
        elif isinstance(sample_mlp_layer, str) and sample_mlp_layer == 'bilinear':
            self.use_bilinear = True
        else:
            raise ValueError(f'{sample_mlp_layer} not supported as mlp_layer arg')

        self.sampler = sampler

        assert gnn_layer > 1
        if gnn_type == 'gine':
            GNNConv = GINEConv
        else:
            raise NotImplementedError

        self.atom_encoder = encoder

        self.n_layers = gnn_layer
        assert 0 <= sample_alpha <= 1, f'sample_alpha should be in [0, 1], got {sample_alpha}'
        self.sample_alpha = sample_alpha

        self.directed_sampling = directed_sampling
        self.use_bn = use_bn
        self.dropout = dropout
        self.use_residual = residual

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList() if use_bn else [None] * gnn_layer
        self.add_mlp_heads = torch.nn.ModuleList()
        self.del_mlp_heads = torch.nn.ModuleList()
        self.intermediate_gnns = torch.nn.ModuleList()

        for i in range(gnn_layer):
            self.convs.append(
                GNNConv(
                    hid_size,
                    Sequential(
                        Linear(hid_size, hid_size),
                        ReLU(),
                        Linear(hid_size, hid_size),
                    ),
                    edge_encoder,
                )
            )
            self.add_mlp_heads.append(
                torch.nn.Bilinear(hid_size, hid_size, ensemble, bias=True) if self.use_bilinear
                else MLP(
                    [hid_size * 2] + [hid_size] * (sample_mlp_layer - 1) + [ensemble],
                    batch_norm=use_bn, dropout=dropout)
            )
            self.del_mlp_heads.append(
                torch.nn.Bilinear(hid_size, hid_size, ensemble, bias=True) if self.use_bilinear
                else MLP(
                    [hid_size * 2] + [hid_size] * (sample_mlp_layer - 1) + [ensemble],
                    batch_norm=use_bn, dropout=dropout)
            )
            if use_bn:
                self.bns.append(BN(hid_size))
            self.intermediate_gnns.append(make_intermediate_gnn())

        # intra-graph pooling
        self.graph_pool_idx = 'batch'
        self.graph_pooling = graph_pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling is None:  # node pred
            self.pool = lambda x, *args: x
        elif graph_pooling == 'transductive':
            self.pool = lambda x, transductive_mask: x[transductive_mask]
            self.graph_pool_idx = 'transductive_mask'
        elif graph_pooling == 'root':
            self.pool = lambda x, root_mask: x[root_mask]
            self.graph_pool_idx = 'root_mask'
        else:
            raise NotImplementedError

        self.mlp = MLP([hid_size] * mlp_layers_intragraph + [num_classes], dropout=0.)

    def forward(self, data: Union[Data, Batch]):
        data, graphs = data.batch, data.list
        assert hasattr(data, 'edge_candidate') and hasattr(data, 'num_edge_candidate')
        x = self.atom_encoder(data)

        # prepare for addition
        edge_rel = torch.hstack([torch.zeros(1, dtype=torch.long, device=x.device),
                                 torch.cumsum(data.nnodes, dim=0)[:-1]])
        edge_candidate_idx = data.edge_candidate + \
                             edge_rel.repeat_interleave(data.num_edge_candidate)[:, None]
        split_idx = data.nnodes.cpu().tolist()

        # prepare for deletion
        edge_index = data.edge_index
        if not self.directed_sampling:
            edge_index = edge_index[:,edge_index[0] <= edge_index[1]]  # self loops included

        for i, conv in enumerate(self.convs):
            # gnn layer
            x_new = conv(x, data.edge_index, data.edge_attr, data.edge_weight)
            if self.use_bn:
                x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            if self.use_residual:
                x = residual(x, x_new)
            else:
                x = x_new

            if self.use_bilinear:
                select_edge_candidates = self.add_mlp_heads[i](x[edge_candidate_idx[:, 0]],
                                                               x[edge_candidate_idx[:, 1]])
                delete_edge_candidates = self.del_mlp_heads[i](x[edge_index[0]],
                                                               x[edge_index[1]])
            else:
                edge_candidates = torch.hstack([x[edge_candidate_idx[:, 0]],
                                                x[edge_candidate_idx[:, 1]]])
                select_edge_candidates = self.add_mlp_heads[i](edge_candidates)

                cur_edges = torch.hstack([x[edge_index[0]], x[edge_index[1]]])
                delete_edge_candidates = self.del_mlp_heads[i](cur_edges)

            data.x = x
            xs = torch.split(x, split_idx)
            for idx, g in enumerate(graphs):
                g.x = xs[idx]

            sample_ratio = self.sample_alpha ** (self.n_layers - (i + 1))

            sampled_data, mask, auxloss = self.sampler(data,
                                                       graphs,
                                                       self.training,
                                                       select_edge_candidates,
                                                       delete_edge_candidates,
                                                       edge_candidate_idx,
                                                       sample_ratio = sample_ratio)

            # graph is still 21 dimensional
            x = self.intermediate_gnns[i](sampled_data)

        x = self.pool(x, getattr(data, self.graph_pool_idx))
        x = self.mlp(x)
        return x

    def reset_parameters(self):
        raise NotImplementedError


class DynamicRewireTransUpstreamGNN(torch.nn.Module):
    def __init__(self,
                 sampler,
                 make_intermediate_gnn,
                 encoder,
                 hid_size,
                 num_heads,
                 gnn_layer,
                 num_classes,
                 dropout,
                 ensemble,
                 mlp_layers_intragraph,
                 graph_pooling,
                 sample_alpha=1):
        super(DynamicRewireTransUpstreamGNN, self).__init__()

        self.sampler = sampler

        assert gnn_layer > 1
        self.atom_encoder = encoder

        self.n_layers = gnn_layer
        assert 0 <= sample_alpha <= 1, f'sample_alpha should be in [0, 1], got {sample_alpha}'
        self.sample_alpha = sample_alpha

        self.tfs = torch.nn.ModuleList()
        self.attns = torch.nn.ModuleList()
        self.intermediate_gnns = torch.nn.ModuleList()

        for i in range(gnn_layer):
            self.tfs.append(
                TransformerLayer(
                    hid_size,
                    num_heads,
                    dropout=dropout,
                    attn_dropout=0.,
                    layer_norm=False,
                    batch_norm=True,
                    use_spectral_norm=True,
                    use_attn=False,
                )
            )
            self.attns.append(
                AttentionLayer(hid_size, hid_size, ensemble, 0., True)
            )
            self.intermediate_gnns.append(make_intermediate_gnn())

        # intra-graph pooling
        self.graph_pool_idx = 'batch'
        self.graph_pooling = graph_pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling is None:  # node pred
            self.pool = lambda x, *args: x
        elif graph_pooling == 'transductive':
            self.pool = lambda x, transductive_mask: x[transductive_mask]
            self.graph_pool_idx = 'transductive_mask'
        elif graph_pooling == 'root':
            self.pool = lambda x, root_mask: x[root_mask]
            self.graph_pool_idx = 'root_mask'
        else:
            raise NotImplementedError

        self.mlp = MLP([hid_size] * mlp_layers_intragraph + [num_classes], dropout=0.)

    def forward(self, data: Union[Data, Batch]):
        data, graphs = data.batch, data.list
        x = self.atom_encoder(data)

        _, real_node_mask = to_dense_batch(x, data.batch)
        real_node_node_mask = torch.einsum('bn,bm->bnm', real_node_mask, real_node_mask)
        split_idx = data.nnodes.cpu().tolist()

        for i, tf in enumerate(self.tfs):
            # gnn layer
            x = tf(x, data)
            x, _ = to_dense_batch(x, data.batch)
            attention_score = self.attns[i](x)[1]
            x = x[real_node_mask]

            # update x embedding with upstream too, thus coupled
            data.x = x
            xs = torch.split(x, split_idx)
            for idx, g in enumerate(graphs):
                g.x = xs[idx]

            sample_ratio = self.sample_alpha ** (self.n_layers - (i + 1))

            sampled_data, mask, auxloss = self.sampler(data,
                                                       graphs,
                                                       self.training,
                                                       attention_score,
                                                       real_node_node_mask,
                                                       sample_ratio=sample_ratio)

            # graph is still 21 dimensional
            x = self.intermediate_gnns[i](sampled_data)

        x = self.pool(x, getattr(data, self.graph_pool_idx))
        x = self.mlp(x)
        return x

    def reset_parameters(self):
        raise NotImplementedError


class DecoupledDynamicRewireGNN(torch.nn.Module):
    def __init__(self,
                 sampler,
                 make_intermediate_gnn,
                 encoder,
                 edge_encoder,
                 hid_size,
                 gnn_type,
                 gnn_layer,
                 sample_mlp_layer,
                 num_classes,
                 directed_sampling,
                 residual,
                 dropout,
                 ensemble,
                 use_bn,
                 mlp_layers_intragraph,
                 graph_pooling,
                 sample_alpha=1,
                 input_from_downstream=False):
        super(DecoupledDynamicRewireGNN, self).__init__()

        if isinstance(sample_mlp_layer, int):
            self.use_bilinear = False
        elif isinstance(sample_mlp_layer, str) and sample_mlp_layer == 'bilinear':
            self.use_bilinear = True
        else:
            raise ValueError(f'{sample_mlp_layer} not supported as mlp_layer arg')

        self.sampler = sampler

        assert gnn_layer > 1
        if gnn_type == 'gine':
            GNNConv = GINEConv
        else:
            raise NotImplementedError

        self.atom_encoder = encoder

        self.n_layers = gnn_layer
        assert 0 <= sample_alpha <= 1, f'sample_alpha should be in [0, 1], got {sample_alpha}'
        self.sample_alpha = sample_alpha
        self.input_from_down = input_from_downstream
        print(f'input from downstream: {self.input_from_down}')

        self.directed_sampling = directed_sampling
        self.use_bn = use_bn
        self.dropout = dropout
        self.use_residual = residual

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList() if use_bn else [None] * gnn_layer
        self.add_mlp_heads = torch.nn.ModuleList()
        self.del_mlp_heads = torch.nn.ModuleList()
        self.intermediate_gnns = torch.nn.ModuleList()

        for i in range(gnn_layer):
            self.convs.append(
                GNNConv(
                    hid_size,
                    Sequential(
                        Linear(hid_size, hid_size),
                        ReLU(),
                        Linear(hid_size, hid_size),
                    ),
                    edge_encoder,
                )
            )
            self.add_mlp_heads.append(
                torch.nn.Bilinear(hid_size, hid_size, ensemble, bias=True) if self.use_bilinear
                else MLP([hid_size * 2] + [hid_size] * (sample_mlp_layer - 1) + [ensemble],
                         batch_norm=use_bn,
                         dropout=dropout)
            )
            self.del_mlp_heads.append(
                torch.nn.Bilinear(hid_size, hid_size, ensemble, bias=True) if self.use_bilinear
                else MLP(
                    [hid_size * 2] + [hid_size] * (sample_mlp_layer - 1) + [ensemble],
                    batch_norm=use_bn,
                    dropout=dropout)
            )
            if use_bn:
                self.bns.append(BN(hid_size))
            self.intermediate_gnns.append(make_intermediate_gnn())

        # intra-graph pooling
        self.graph_pool_idx = 'batch'
        self.graph_pooling = graph_pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling is None:  # node pred
            self.pool = lambda x, *args: x
        elif graph_pooling == 'transductive':
            self.pool = lambda x, transductive_mask: x[transductive_mask]
            self.graph_pool_idx = 'transductive_mask'
        elif graph_pooling == 'root':
            self.pool = lambda x, root_mask: x[root_mask]
            self.graph_pool_idx = 'root_mask'
        else:
            raise NotImplementedError

        self.mlp = MLP([hid_size] * mlp_layers_intragraph + [num_classes], dropout=0.)

    def forward(self, data: Union[Data, Batch]):
        data, graphs = data.batch, data.list
        assert hasattr(data, 'edge_candidate') and hasattr(data, 'num_edge_candidate')
        x = self.atom_encoder(data)

        # prepare for addition
        edge_rel = torch.hstack([torch.zeros(1, dtype=torch.long, device=x.device),
                                 torch.cumsum(data.nnodes, dim=0)[:-1]])
        edge_candidate_idx = data.edge_candidate + \
                             edge_rel.repeat_interleave(data.num_edge_candidate)[:, None]
        split_idx = data.nnodes.cpu().tolist()

        # prepare for deletion
        edge_index = data.edge_index
        if not self.directed_sampling:
            edge_index = edge_index[:,edge_index[0] <= edge_index[1]]  # self loops included

        x_down = x.clone()

        if not self.input_from_down:
            x_up = x
            
        for i, conv in enumerate(self.convs):
            # gnn layer
            if self.input_from_down:
                x_up = x_down#.clone().detach()

            x_up_new = conv(x_up, data.edge_index, data.edge_attr, data.edge_weight)

            if self.use_bilinear:
                select_edge_candidates = self.add_mlp_heads[i](x_up_new[edge_candidate_idx[:, 0]],
                                                               x_up_new[edge_candidate_idx[:, 1]])
                delete_edge_candidates = self.del_mlp_heads[i](x_up_new[edge_index[0]],
                                                               x_up_new[edge_index[1]])
            else:
                edge_candidates = torch.hstack([x_up_new[edge_candidate_idx[:, 0]],
                                                x_up_new[edge_candidate_idx[:, 1]]])
                select_edge_candidates = self.add_mlp_heads[i](edge_candidates)

                cur_edges = torch.hstack([x_up_new[edge_index[0]],
                                          x_up_new[edge_index[1]]])
                delete_edge_candidates = self.del_mlp_heads[i](cur_edges)

            if self.use_bn:
                x_up_new = self.bns[i](x_up_new)
            x_up_new = F.relu(x_up_new)
            x_up_new = F.dropout(x_up_new, p=self.dropout, training=self.training)
            if self.use_residual:
                x_up = residual(x_up, x_up_new)
            else:
                x_up = x_up_new

            data.x = x_down
            xs = torch.split(x_down, split_idx)
            for idx, g in enumerate(graphs):
                g.x = xs[idx]

            sample_ratio = self.sample_alpha ** (self.n_layers - (i + 1))

            sampled_data, mask, auxloss = self.sampler(data,
                                                       graphs,
                                                       self.training,
                                                       select_edge_candidates,
                                                       delete_edge_candidates,
                                                       edge_candidate_idx,
                                                       sample_ratio = sample_ratio)

            x_down_new = self.intermediate_gnns[i](sampled_data)
            x_down_new = F.dropout(F.relu(x_down_new), p=self.dropout, training=self.training)
            if self.use_residual:
                x_down = residual(x_down, x_down_new)
            else:
                x_down = x_down_new

        x = self.pool(x_down, getattr(data, self.graph_pool_idx))
        x = self.mlp(x)
        return x

    def reset_parameters(self):
        raise NotImplementedError


class DecoupledDynamicRewireTransUpstreamGNN(torch.nn.Module):
    def __init__(self,
                 sampler,
                 make_intermediate_gnn,
                 encoder,
                 hid_size,
                 gnn_layer,
                 num_classes,
                 dropout,
                 num_heads,
                 ensemble,
                 mlp_layers_intragraph,
                 graph_pooling,
                 sample_alpha=1,
                 input_from_downstream=False):
        super(DecoupledDynamicRewireTransUpstreamGNN, self).__init__()

        self.sampler = sampler
        assert gnn_layer > 1
        self.atom_encoder = encoder
        self.dropout = dropout

        self.n_layers = gnn_layer
        assert 0 <= sample_alpha <= 1, f'sample_alpha should be in [0, 1], got {sample_alpha}'
        self.sample_alpha = sample_alpha
        self.input_from_down = input_from_downstream
        print(f'input from downstream: {self.input_from_down}')

        self.tfs = torch.nn.ModuleList()
        self.attns = torch.nn.ModuleList()
        self.intermediate_gnns = torch.nn.ModuleList()

        for i in range(gnn_layer):
            self.tfs.append(
                TransformerLayer(
                    hid_size,
                    num_heads,
                    dropout=dropout,
                    attn_dropout=0.,
                    layer_norm=False,
                    batch_norm=True,
                    use_spectral_norm=True,
                    use_attn=False,
                )
            )
            self.attns.append(
                AttentionLayer(hid_size, hid_size, ensemble, 0., True)
            )
            self.intermediate_gnns.append(make_intermediate_gnn())

        # intra-graph pooling
        self.graph_pool_idx = 'batch'
        self.graph_pooling = graph_pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling is None:  # node pred
            self.pool = lambda x, *args: x
        elif graph_pooling == 'transductive':
            self.pool = lambda x, transductive_mask: x[transductive_mask]
            self.graph_pool_idx = 'transductive_mask'
        elif graph_pooling == 'root':
            self.pool = lambda x, root_mask: x[root_mask]
            self.graph_pool_idx = 'root_mask'
        else:
            raise NotImplementedError

        self.mlp = MLP([hid_size] * mlp_layers_intragraph + [num_classes], dropout=0.)

    def forward(self, data: Union[Data, Batch]):
        data, graphs = data.batch, data.list
        x = self.atom_encoder(data)

        _, real_node_mask = to_dense_batch(x, data.batch)
        real_node_node_mask = torch.einsum('bn,bm->bnm', real_node_mask, real_node_mask)
        split_idx = data.nnodes.cpu().tolist()

        x_down = x.clone()

        if not self.input_from_down:
            x_up = x

        for i, tf in enumerate(self.tfs):
            # gnn layer
            if self.input_from_down:
                x_up = x_down.clone().detach()

            x_up = tf(x_up, data)
            x_up, _ = to_dense_batch(x_up, data.batch)
            attention_score = self.attns[i](x_up)[1]
            x_up = x_up[real_node_mask]

            data.x = x_down
            xs = torch.split(x_down, split_idx)
            for idx, g in enumerate(graphs):
                g.x = xs[idx]

            sample_ratio = self.sample_alpha ** (self.n_layers - (i + 1))

            sampled_data, mask, auxloss = self.sampler(data,
                                                       graphs,
                                                       self.training,
                                                       attention_score,
                                                       real_node_node_mask,
                                                       sample_ratio=sample_ratio)

            x_down_new = self.intermediate_gnns[i](sampled_data)
            x_down_new = F.dropout(F.relu(x_down_new), p=self.dropout,
                                   training=self.training)
            x_down = residual(x_down, x_down_new)

        x = self.pool(x_down, getattr(data, self.graph_pool_idx))
        x = self.mlp(x)
        return x

    def reset_parameters(self):
        raise NotImplementedError
