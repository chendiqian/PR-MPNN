from ortools.linear_solver import pywraplp
import torch


# for attributes see pywrapper
# https://github.com/google/or-tools/blob/stable/ortools/linear_solver/python/linear_solver.i#L376
def get_personalized_problem(seed_node, value_list, solver, x):
    n_nodes = len(value_list)

    # obj
    objective = solver.Objective()
    for i in range(n_nodes):
        objective.SetCoefficient(x[i], value_list[i])
    objective.SetMaximization()

    # personalized seed node constraint
    solver.constraints()[-1].Clear()
    solver.constraints()[-1].SetCoefficient(x[seed_node], 1.)

    return solver


def get_basic_problem(n_nodes, edge_index_lst, nodes_per_node):
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # variable
    x = [solver.IntVar(0, 1, f'x[{i}]') for i in range(n_nodes)]
    e = {}
    for i, j in edge_index_lst:
        e[f'{i}_{j}'] = solver.IntVar(0, 1, f'e[{i}][{j}]')

    # nodes per node
    solver.Add(sum(x) == nodes_per_node)

    # edge & nodes
    for i, j in edge_index_lst:
        solver.Add(e[f'{i}_{j}'] <= x[i])
        solver.Add(e[f'{i}_{j}'] <= x[j])
        solver.Add(e[f'{i}_{j}'] >= (x[i] + x[j] - 1))

    # connectedness
    solver.Add(sum(e.values()) >= nodes_per_node - 1)

    # the seed node must be chosen
    solver.Add(x[0] >= 1)

    return solver, x, e


def get_or_optim_subgraphs(edge_index: torch.Tensor, value_tensor: torch.Tensor, nodes_per_node: int) -> torch.Tensor:
    n_nodes, _, channels = value_tensor.shape

    if nodes_per_node >= n_nodes:
        return torch.ones_like(value_tensor, dtype=torch.float32, device=value_tensor.device)

    value_list = torch.permute(value_tensor, (2, 0, 1)).cpu().tolist()
    directed_mask = edge_index[0] < edge_index[1]
    edge_index_lst = edge_index.t()[directed_mask].cpu().tolist()

    solver, x, e = get_basic_problem(n_nodes, edge_index_lst, nodes_per_node)
    solution = torch.empty(n_nodes, n_nodes, channels, dtype=torch.float32, device=edge_index.device)
    for c in range(channels):
        for i in range(n_nodes):
            solver = get_personalized_problem(i, value_list[c][i], solver, x)
            status = solver.Solve()

            if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
                raise ValueError("No solution")
            for j in range(n_nodes):
                solution[i, j, c] = x[j].solution_value()

    return solution
