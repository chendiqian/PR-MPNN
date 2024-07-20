from typing import List, Tuple

import torch
from ml_collections import ConfigDict
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch, to_undirected
from torch_scatter import scatter

from models.rewire_utils import batch_repeat_edge_index, sparsify_edge_weight_simplified
from models.aux_loss import get_auxloss
from models.rewirer import GraphRewirer
from simple.simple_scheme import EdgeSIMPLEBatched

LARGE_NUMBER = 1.e10


class SimplifiedGraphRewirer(GraphRewirer):
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
                add_edge_weight)
            rewired_batch = [rewired_batch]  # return as a list
        else:
            if add_edge_weight is not None:
                rewired_add_batch = self.merge_add(
                    dumb_repeat_batch,
                    add_edge_index,
                    add_edge_weight)
                rewired_batch = [rewired_add_batch]
            else:
                rewired_batch = []

            if del_edge_weight is not None:
                rewired_del_batch = self.merge_del(new_graphs, del_edge_weight)
                rewired_batch.append(rewired_del_batch)
        return rewired_batch, auxloss1 + auxloss2

    def merge_del_add(self,
                      rewired_batch: Batch,
                      add_edge_index: torch.LongTensor,
                      del_edge_weight: torch.FloatTensor,
                      add_edge_weight: torch.FloatTensor, *args, ** kwargs):
        merged_edge_index = torch.cat([rewired_batch.edge_index, add_edge_index], dim=1)
        merged_edge_weight = torch.cat([del_edge_weight, add_edge_weight], dim=-1)
        if rewired_batch.edge_attr is not None:
            merged_edge_attr = torch.cat([rewired_batch.edge_attr,
                                          rewired_batch.edge_attr.new_zeros(add_edge_weight.shape[-1],
                                                                            rewired_batch.edge_attr.shape[1])], dim=0)
        else:
            merged_edge_attr = None

        rewired_batch.edge_index = merged_edge_index
        rewired_batch.edge_attr = merged_edge_attr
        rewired_batch.edge_weight = merged_edge_weight
        rewired_batch = sparsify_edge_weight_simplified(rewired_batch, self.training)
        return rewired_batch

    def merge_add(self,
                  rewired_batch: Batch,
                  add_edge_index: torch.LongTensor,
                  add_edge_weight: torch.FloatTensor, *args, ** kwargs):
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
        del_rewired_batch.edge_weight = del_edge_weight
        del_rewired_batch = sparsify_edge_weight_simplified(del_rewired_batch, self.training)
        return del_rewired_batch
