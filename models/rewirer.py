from typing import List, Tuple, Optional

import torch
from ml_collections import ConfigDict
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch, to_undirected
from torch_scatter import scatter

from models.rewire_utils import (non_merge_coalesce,
                                 batch_repeat_edge_index,
                                 sparsify_edge_weight,
                                 sparsify_edge_weight_simplified)
from models.aux_loss import get_auxloss
from simple.simple_scheme import EdgeSIMPLEBatched

LARGE_NUMBER = 1.e10


class GraphRewirer(torch.nn.Module):
    def __init__(self,
                 add_k: int,
                 del_k: int,
                 train_ensemble: int,
                 val_ensemble: int,
                 sampler: EdgeSIMPLEBatched,
                 separate: bool = False,
                 directed_sampling: bool = False,
                 auxloss_dict: ConfigDict = None,
                 data_list_compatible: bool = False):
        """
        Especially, data_list_compatible is an important option
        We recommend False by default for efficiency
        However, if you wish to de-collate the data batch into separate graphs
        You need to deal with _inc_dict and _slice_dict properly, and set it True
        """
        super(GraphRewirer, self).__init__()
        assert del_k > 0 or add_k > 0

        self.add_k = add_k
        self.del_k = del_k
        self.train_ensemble = train_ensemble
        self.val_ensemble = val_ensemble
        self.sampler = sampler
        self.separate = separate
        self.directed_sampling = directed_sampling
        self.auxloss_dict = auxloss_dict
        self.data_list_compatible = data_list_compatible

    def forward(self,
                dat_batch: Data,
                addition_logits: torch.Tensor,
                deletion_logits: torch.Tensor,
                edge_candidate_idx: torch.Tensor) -> Tuple[List[Batch], float]:
        device = addition_logits.device

        graphs = Batch.to_data_list(dat_batch)

        edge_ptr = dat_batch._slice_dict['edge_index'].to(device)
        nedges = edge_ptr[1:] - edge_ptr[:-1]

        # number of sampling from EACH ensemble
        VE = self.train_ensemble if self.training else self.val_ensemble
        # number of ensemble given by upstream model
        E = addition_logits.shape[-1]

        add_edge_weight, add_edge_index, auxloss1 = self.add_edge(dat_batch, addition_logits, edge_candidate_idx)
        del_edge_weight, auxloss2 = self.del_edge(dat_batch, deletion_logits, nedges)

        new_graphs = graphs * (E * VE)
        dumb_repeat_batch = Batch.from_data_list(new_graphs)

        # del and add are modified on the same graph, in-place
        if not self.separate and del_edge_weight is not None and add_edge_weight is not None:
            rewired_batch = self.merge_del_add(
                dumb_repeat_batch,
                add_edge_index,
                del_edge_weight,
                add_edge_weight,
                nedges.repeat(E * VE) if self.data_list_compatible else None,
                (dat_batch.num_edge_candidate * (2 if not self.directed_sampling else 1)).repeat(E * VE)
                if self.data_list_compatible else None)
            rewired_batch = [rewired_batch]  # return as a list
        else:
            if add_edge_weight is not None:
                rewired_add_batch = self.merge_add(
                    dumb_repeat_batch,
                    add_edge_index,
                    add_edge_weight,
                    nedges.repeat(E * VE) if self.data_list_compatible else None,
                    (dat_batch.num_edge_candidate * (2 if not self.directed_sampling else 1)).repeat(E * VE)
                    if self.data_list_compatible else None)
                rewired_batch = [rewired_add_batch]
            else:
                rewired_batch = []

            if del_edge_weight is not None:
                rewired_del_batch = self.merge_del(new_graphs, del_edge_weight)
                rewired_batch.append(rewired_del_batch)
        return rewired_batch, auxloss1 + auxloss2

    def add_edge(self,
                 dat_batch: Data,
                 addition_logits: torch.FloatTensor,
                 edge_candidate_idx: torch.LongTensor):

        device = addition_logits.device

        # number of sampling from EACH ensemble
        VE = self.train_ensemble if self.training else self.val_ensemble
        # number of ensemble given by upstream model
        E = addition_logits.shape[-1]

        auxloss = 0.
        if self.add_k > 0:
            # nnedge_candid * ensemble -> batchsize * Max_nnedge_candid * ensemble
            # mask: batchsize * Max_nnedge_candid
            batch_addition_logits, real_node_mask = to_dense_batch(addition_logits,
                                                                   torch.arange(dat_batch.num_graphs,
                                                                                device=device).repeat_interleave(
                                                                       dat_batch.num_edge_candidate),
                                                                   max_num_nodes=dat_batch.num_edge_candidate.max())

            padding_bias = (~real_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
            biased_addition_logits = batch_addition_logits - padding_bias

            # (#sampled, B, Nmax, E), (B, Nmax, E)
            if self.training:
                node_mask, marginals = self.sampler(biased_addition_logits, self.add_k, self.train_ensemble)
                auxloss = get_auxloss(self.auxloss_dict, batch_addition_logits, node_mask)
            else:
                node_mask, marginals = self.sampler.validation(biased_addition_logits, self.add_k, self.val_ensemble)

            sampled_edge_weights = torch.stack([marginals] * VE, dim=0)
            if not self.training:
                sampled_edge_weights = sampled_edge_weights * node_mask

            # num_edges x E x VE
            add_edge_weight = sampled_edge_weights.permute((1, 2, 3, 0))[real_node_mask].reshape(-1, E * VE)
            add_edge_index = edge_candidate_idx.T

            if not self.directed_sampling:
                add_edge_index, add_edge_weight = to_undirected(add_edge_index,
                                                                add_edge_weight,
                                                                num_nodes=dat_batch.num_nodes)

            add_edge_index = batch_repeat_edge_index(add_edge_index, dat_batch.num_nodes, E * VE)
            add_edge_weight = add_edge_weight.t().reshape(-1)
        else:
            add_edge_weight = None
            add_edge_index = None
        return add_edge_weight, add_edge_index, auxloss

    def del_edge(self,
                 dat_batch: Data,
                 deletion_logits: torch.FloatTensor,
                 nedges: torch.LongTensor):
        device = deletion_logits.device

        auxloss = 0.
        if self.del_k > 0:
            batch_idx = torch.arange(dat_batch.num_graphs, device=device).repeat_interleave(nedges)
            if self.directed_sampling:
                batch_deletion_logits, real_node_mask = to_dense_batch(deletion_logits,
                                                                       batch_idx,
                                                                       max_num_nodes=nedges.max())
            else:
                direct_mask = dat_batch.edge_index[0] <= dat_batch.edge_index[1]
                directed_edge_index = dat_batch.edge_index[:, direct_mask]
                num_direct_edges = scatter(direct_mask.long(),
                                           batch_idx,
                                           reduce='sum',
                                           dim_size=dat_batch.num_graphs)
                batch_deletion_logits, real_node_mask = to_dense_batch(deletion_logits,
                                                                       batch_idx[direct_mask],
                                                                       max_num_nodes=num_direct_edges.max())

            # we select the least scores
            batch_deletion_logits = -batch_deletion_logits
            padding_bias = (~real_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
            bias_deletion_logits = batch_deletion_logits - padding_bias

            # (#sampled, B, Nmax, E), (B, Nmax, E)
            if self.training:
                node_mask, marginals = self.sampler(bias_deletion_logits, self.del_k, self.train_ensemble)
                auxloss = get_auxloss(self.auxloss_dict, batch_deletion_logits, node_mask)
            else:
                node_mask, marginals = self.sampler.validation(bias_deletion_logits, self.del_k, self.val_ensemble)

            VE, B, N, E = node_mask.shape
            sampled_edge_weights = 1. - node_mask

            # num_edges x E x VE
            del_edge_weight = sampled_edge_weights.permute((1, 2, 3, 0))[real_node_mask].reshape(-1, E * VE)
            if not self.directed_sampling:
                # reduce must be mean, otherwise the self loops have double weights
                _, del_edge_weight = to_undirected(directed_edge_index, del_edge_weight,
                                                   num_nodes=dat_batch.num_nodes,
                                                   reduce='mean')
            del_edge_weight = del_edge_weight.t().reshape(-1)
        else:
            del_edge_weight = None

        return del_edge_weight, auxloss

    def merge_del_add(self,
                      rewired_batch: Batch,
                      add_edge_index: torch.LongTensor,
                      del_edge_weight: torch.FloatTensor,
                      add_edge_weight: torch.FloatTensor,
                      original_num_edges: Optional[torch.LongTensor],
                      new_num_edges: Optional[torch.LongTensor]):
        merged_edge_index = torch.cat([rewired_batch.edge_index, add_edge_index], dim=1)
        merged_edge_weight = torch.cat([del_edge_weight, add_edge_weight], dim=-1)
        if rewired_batch.edge_attr is not None:
            merged_edge_attr = torch.cat([rewired_batch.edge_attr,
                                          rewired_batch.edge_attr.new_zeros(add_edge_weight.shape[-1],
                                                                            rewired_batch.edge_attr.shape[1])], dim=0)
        else:
            merged_edge_attr = None

        if self.data_list_compatible:
            # pyg coalesce force to merge duplicate edges, which is in conflict with our _slice_dict calculation
            merged_edge_index, merged_edge_attr, merged_edge_weight = non_merge_coalesce(
                edge_index=merged_edge_index,
                edge_attr=merged_edge_attr,
                edge_weight=merged_edge_weight,
                num_nodes=rewired_batch.num_nodes)
            rewired_batch.edge_index = merged_edge_index
            rewired_batch.edge_attr = merged_edge_attr

            # inc dict
            rewired_batch._slice_dict['edge_index'] = torch.hstack([add_edge_index.new_zeros(1),
                                                                    (original_num_edges + new_num_edges).cumsum(dim=0)])

            rewired_batch = sparsify_edge_weight(rewired_batch, merged_edge_weight, self.training)
        else:
            rewired_batch.edge_index = merged_edge_index
            rewired_batch.edge_attr = merged_edge_attr
            rewired_batch.edge_weight = merged_edge_weight
            rewired_batch = sparsify_edge_weight_simplified(rewired_batch, self.training)
        return rewired_batch

    def merge_add(self,
                  rewired_batch: Batch,
                  add_edge_index: torch.LongTensor,
                  add_edge_weight: torch.FloatTensor,
                  original_num_edges: Optional[torch.LongTensor],
                  new_num_edges: Optional[torch.LongTensor]):
        dumb_repeat_edge_index = rewired_batch.edge_index
        merged_edge_index = torch.cat([dumb_repeat_edge_index, add_edge_index], dim=1)
        merged_edge_weight = torch.cat(
            [add_edge_weight.new_ones(dumb_repeat_edge_index.shape[1]), add_edge_weight], dim=-1)
        if rewired_batch.edge_attr is not None:
            merged_edge_attr = torch.cat([rewired_batch.edge_attr,
                                          rewired_batch.edge_attr.new_zeros(
                                              add_edge_weight.shape[-1],
                                              rewired_batch.edge_attr.shape[1])], dim=0)
        else:
            merged_edge_attr = None

        if self.data_list_compatible:
            # pyg coalesce force to merge duplicate edges, which is in conflict with our _slice_dict calculation
            merged_edge_index, merged_edge_attr, merged_edge_weight = non_merge_coalesce(
                edge_index=merged_edge_index,
                edge_attr=merged_edge_attr,
                edge_weight=merged_edge_weight,
                num_nodes=rewired_batch.num_nodes)
            rewired_batch.edge_index = merged_edge_index
            rewired_batch.edge_attr = merged_edge_attr

            # inc dict
            rewired_batch._slice_dict['edge_index'] = torch.hstack([add_edge_index.new_zeros(1),
                                                                    (original_num_edges + new_num_edges).cumsum(
                                                                        dim=0)])
            rewired_batch = sparsify_edge_weight(rewired_batch, merged_edge_weight, self.training)
        else:
            rewired_batch.edge_index = merged_edge_index
            rewired_batch.edge_attr = merged_edge_attr
            rewired_batch.edge_weight = merged_edge_weight
            rewired_batch = sparsify_edge_weight_simplified(rewired_batch, self.training)
        return rewired_batch

    def merge_del(self,
                  new_graphs: List[Data],
                  del_edge_weight: torch.FloatTensor):
        # cannot modify the `rewired_batch`, it is a mutable object
        # also we normally add edges, so it is already modified
        # so we batch a new batch
        del_rewired_batch = Batch.from_data_list(new_graphs)

        if self.data_list_compatible:
            del_rewired_batch = sparsify_edge_weight(del_rewired_batch, del_edge_weight, self.training)
        else:
            del_rewired_batch.edge_weight = del_edge_weight
            del_rewired_batch = sparsify_edge_weight_simplified(del_rewired_batch, self.training)
        return del_rewired_batch
