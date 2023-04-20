import torch


def residual(y_old: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
    if y_old.shape == y_new.shape:
        return (y_old + y_new) / 2 ** 0.5
    else:
        return y_new


def reset_sequential_parameters(seq: torch.nn.Sequential) -> None:
    lst = torch.nn.ModuleList(seq)
    for l in lst:
        if not isinstance(l, (torch.nn.ReLU, torch.nn.Dropout, torch.nn.GELU)):
            l.reset_parameters()
    # return torch.nn.Sequential(**lst)


def reset_modulelist_parameters(seq: torch.nn.ModuleList) -> None:
    for l in seq:
        if not isinstance(l, torch.nn.ReLU):
            l.reset_parameters()


def cat_pooling(x, graph_idx):
    """

    Args:
        x:
        graph_idx: should be like (0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3) in case of 4 graphs, 3 ensembles

    Returns:

    """
    unique, ensemble = torch.unique(graph_idx, return_counts=True)
    assert len(torch.unique(ensemble)) == 1, "each graph should have the same number of ensemble!"
    n_ensemble = ensemble[0]
    n_graphs = len(unique)

    new_x = x.new_empty(n_graphs, n_ensemble, x.shape[-1])
    ensemble_idx = torch.arange(n_ensemble, device=x.device).repeat_interleave(n_graphs)
    new_x[graph_idx, ensemble_idx, :] = x
    return new_x.reshape(n_graphs, -1)
