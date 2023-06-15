import os
import pickle
from collections import defaultdict
from typing import List

import torch
import torch._dynamo
torch._dynamo.reset()
from simple.create_simple_constraint import create_and_save

DISABLE = False
MODE = 'default'


@torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
def levelwiseSL(levels: List[torch.Tensor], idx2primesub: torch.Tensor,
                data: torch.Tensor, theta: torch.Tensor):
    for level in levels:
        theta[level] = data[idx2primesub[level]].sum(-2)
        data[level] = theta[level].logsumexp(-2)
        theta[level] -= data[level].unsqueeze(1)
    return data[levels[-1]]


@torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
def levelwiseMars(levels: List[torch.Tensor], idx2primesub: torch.Tensor,
                  data: torch.Tensor, theta: torch.Tensor, parents: torch.Tensor):
    for level in reversed(levels):
        data[level] = (theta[parents[level].unbind(-1)] + data[
            parents[level].unbind(-1)[0]]).logsumexp(-2)


@torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
def log1mexp(x):
    # Source: https://github.com/wouterkool/estimating-gradients-without-replacement/blob/9d8bf8b/bernoulli/gumbel.py#L7-L11
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    x = torch.where(
        x > -0.6931471805599453094,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )

    return x


def levelOrder(beta):
    """
    :type root: Node
    :rtype: List[List[int]]
    """
    seen = dict()
    nodes = [beta]
    level = []
    answer = []
    result = [[beta]]
    while len(nodes) != 0:
        for a in nodes:
            if not a.is_decomposition():
                continue
            for element in a.elements:
                for e in element:
                    if not e.is_decomposition():
                        continue
                    if seen.get(e) != None: continue
                    seen[e] = True
                    level.append(e)
        nodes = level
        for i in level:
            answer.append(i)
        level = []
        answer = list(dict.fromkeys(answer))
        result.append(answer)
        answer = []
    return result[:-1]


@torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
def gumbel_keys(w, time_sampled):
    # sample some gumbels
    uniform = torch.rand((time_sampled,) + w.shape, device=w.device)  # .to(device)
    z = -torch.log(-torch.log(uniform))
    w = w + z
    return w


@torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
def sample_subset(w, k, time_sampled):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
    '''
    with torch.no_grad():
        w = gumbel_keys(w, time_sampled)
        return w.topk(k, dim=-1).indices


class Layer:
    def __init__(self, n, k, device, root='./simple_configs'):

        if not os.path.isdir(root):
            os.mkdir(root)

        if not os.path.isfile(f'{root}/{n}C{k}.pkl'):
            create_and_save(n, k, root)
        with open(f'{root}/{n}C{k}.pkl', 'rb') as inp:
            beta = pickle.load(inp)

        max_elements = 0
        for node in beta.positive_iter():
            if node.is_decomposition():
                max_elements = max(max_elements, len(node.elements))

        levels_nodes = levelOrder(beta)

        # Reset ids
        nodes = [node for node in beta.positive_iter()]
        nodes = list(dict.fromkeys(nodes))

        id = 0
        for e in nodes:
            e.id = id
            id += 1
        self.id = id

        parents_dict = defaultdict(list)
        for node in beta.positive_iter():
            if node.is_decomposition():
                for i, (p, s) in enumerate(node.elements):
                    parents_dict[p.id] += [[node.id, i]]
                    parents_dict[s.id] += [[node.id, i]]

        # Set up the parents for an efficient backward pass
        max_parents = 0
        for p in parents_dict.values():
            max_parents = max(len(p), max_parents)

        parents = torch.empty((id, max_parents, 2), dtype=torch.int, device=device).fill_(id)
        for k, v in parents_dict.items():
            parents[k] = torch.tensor(v + [[id, 0]] * (max_parents - len(v)),
                                      dtype=torch.int, device=device)  # .to(device)
            # parents[k] = torch.nn.functional.pad(tmp, (0,0,0,max_parents - len(tmp)), value=id)
        self.parents = parents

        # Levels
        levels = []
        for level in levels_nodes:
            levels.append(torch.tensor([l.id for l in level], dtype=torch.int, device=device))
        levels.reverse()
        self.levels = levels

        # true indices
        true_indices = torch.tensor([node.id for node in nodes if node.is_true()], dtype=torch.int, device=device)
        self.true_indices = true_indices

        # Literal indices
        literal_indices = torch.tensor(
            [[node.id, node.literal] for node in nodes if node.is_literal()],
            dtype=torch.int, device=device)
        literal_indices, literal_mask = literal_indices.unbind(-1)
        literal_mask = literal_mask.abs() - 1, (literal_mask > 0).long()
        self.literal_indices = literal_indices
        self.literal_mask = literal_mask

        order = self.literal_mask[0][self.literal_mask[1].bool()].sort().indices
        self.pos_literals = self.literal_indices[self.literal_mask[1].bool()][order]

        # Map nodes to their primes/subs
        idx2primesub = torch.zeros((id, max_elements, 2), dtype=torch.int, device=device)
        for node in nodes:
            if node.is_decomposition():
                tmp = torch.tensor([[p.id, s.id] for p, s in node.elements],
                                   dtype=torch.int)
                idx2primesub[node.id] = torch.nn.functional.pad(tmp, (
                0, 0, 0, max_elements - len(tmp)), value=id)
        self.idx2primesub = idx2primesub

    def __call__(self, log_probs, k):
        samples = self.sample(log_probs, k)
        marginals = self.log_pr(log_probs).exp().permute(1, 0)
        return (samples - marginals).detach() + marginals, marginals

    # @torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
    def log_pr(self, log_probs):
        lit_weights = torch.stack((log1mexp(-log_probs.detach()), log_probs), dim=-1).permute(1, 2, 0)

        data = torch.empty(self.id + 1, log_probs.size(0), device=log_probs.device)
        theta = torch.zeros(self.id + 1, self.idx2primesub.size(1), log_probs.size(0), device=log_probs.device)

        data[self.true_indices] = 0
        data[self.id] = -float(1000)
        data[self.literal_indices] = lit_weights[self.literal_mask[0], self.literal_mask[1]]

        # import pdb; pdb.set_trace()
        res = levelwiseSL(self.levels, self.idx2primesub, data, theta)
        data[self.levels[-1]] -= data[self.levels[-1]]
        levelwiseMars([self.literal_indices] + self.levels[:-1], self.idx2primesub, data, theta, self.parents)

        return data[self.pos_literals]

    @torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
    def sample(self, lit_weights, k, time_sampled = 1):
        with torch.no_grad():
            samples = sample_subset(lit_weights, k, time_sampled)
            samples_hot = lit_weights.new_zeros((time_sampled,) + lit_weights.shape)
            samples_hot.scatter_(2, samples, 1)
            return samples_hot.float()
