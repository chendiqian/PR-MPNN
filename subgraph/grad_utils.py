import torch
from torch_scatter import scatter


class CenterNodeIdentityMapping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, *args):
        """
        Given node mask, return identity, direct grad on the mask

        :param ctx:
        :param mask:
        :return:
        """
        batch_nnodes, new_batch_nnodes = args
        center_mask = centralize(mask, batch_nnodes, new_batch_nnodes)
        return center_mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def centralize(mask: torch.Tensor, batch_nnodes: torch.Tensor, new_batch_nnodes: torch.Tensor):
    """
    Given a node mask, return only the central nodes

    :param mask:
    :param batch_nnodes:
    :param new_batch_nnodes:
    :return:
    """
    subgraphs_seed_nodes = torch.cat([torch.arange(n, device=batch_nnodes.device) for n in batch_nnodes], dim=0)
    subgraphs_seed_nodes[1:] += torch.cumsum(new_batch_nnodes, dim=0)[:-1]
    center_mask = torch.zeros_like(mask, dtype=mask.dtype, device=mask.device)
    center_mask[subgraphs_seed_nodes, :] = 1.
    return center_mask


class Nodemask2Edgemask(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mask, *args):
        """
        Given node masks, return edge masks as edge weights for message passing.

        :param ctx:
        :param mask:
        :param args:
        :return:
        """
        assert mask.dtype == torch.float  # must be differentiable
        edge_index, n_nodes = args
        ctx.save_for_backward(mask, edge_index[1], n_nodes)
        return nodemask2edgemask(mask, edge_index)

    @staticmethod
    def backward(ctx, grad_output):
        _, edge_index_col, n_nodes = ctx.saved_tensors
        final_grad = scatter(grad_output, edge_index_col, dim=0, reduce='mean', dim_size=n_nodes)
        return final_grad, None, None


def nodemask2edgemask(mask: torch.Tensor, edge_index: torch.Tensor, placeholder=None) -> torch.Tensor:
    """
    util function without grad

    :param mask:
    :param edge_index:
    :param placeholder:
    :return:
    """
    return mask[edge_index[0], :] * mask[edge_index[1], :]
