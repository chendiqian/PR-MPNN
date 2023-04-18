# credits to https://github.com/tech-srl/bottleneck/

import itertools
import math
import random

import numpy as np
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset


# https://github.com/tech-srl/bottleneck/blob/main/tasks/tree_dataset.py
class TreeDataset(object):
    def __init__(self, depth):
        super(TreeDataset, self).__init__()
        self.depth = depth
        self.num_nodes, self.edges, self.leaf_indices = self._create_blank_tree()

    def add_child_edges(self, cur_node, max_node):
        edges = []
        leaf_indices = []
        stack = [(cur_node, max_node)]
        while len(stack) > 0:
            cur_node, max_node = stack.pop()
            if cur_node == max_node:
                leaf_indices.append(cur_node)
                continue
            left_child = cur_node + 1
            right_child = cur_node + 1 + ((max_node - cur_node) // 2)
            edges.append([left_child, cur_node])
            edges.append([right_child, cur_node])
            stack.append((right_child, max_node))
            stack.append((left_child, right_child - 1))
        return edges, leaf_indices

    def _create_blank_tree(self):
        max_node_id = 2 ** (self.depth + 1) - 2
        edges, leaf_indices = self.add_child_edges(cur_node=0, max_node=max_node_id)
        return max_node_id + 1, edges, leaf_indices

    def create_blank_tree(self, add_self_loops=True):
        edge_index = torch.tensor(self.edges).t()
        if add_self_loops:
            edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index=edge_index, )
        return edge_index

    def generate_data(self, train_fraction):
        data_list = []

        for comb in self.get_combinations():
            edge_index = self.create_blank_tree(add_self_loops=True)
            nodes = torch.tensor(self.get_nodes_features(comb), dtype=torch.long)
            root_mask = torch.tensor([True] + [False] * (len(nodes) - 1))
            label = self.label(comb)
            data_list.append(Data(x=nodes, edge_index=edge_index, root_mask=root_mask, y=label))

        dim0, out_dim = self.get_dims()
        X_train, X_test = train_test_split(
            data_list, train_size=train_fraction, shuffle=True, stratify=[data.y for data in data_list])


        return X_train, X_test, dim0, out_dim

    # Every sub-class should implement the following methods:
    def get_combinations(self):
        raise NotImplementedError

    def get_nodes_features(self, combination):
        raise NotImplementedError

    def label(self, combination):
        raise NotImplementedError

    def get_dims(self):
        raise NotImplementedError


# https://github.com/tech-srl/bottleneck/blob/bfe83b4a6dd7939ddb19cabea4f1e072f3c35432/tasks/dictionary_lookup.py#L10
class DictionaryLookupDataset(TreeDataset):
    def __init__(self, depth, seed):
        super(DictionaryLookupDataset, self).__init__(depth)
        random.seed(seed)
        np.random.seed(seed)

    def get_combinations(self):
        # returns: an iterable of [key, permutation(leaves)]
        # number of combinations: (num_leaves!)*num_choices
        num_leaves = len(self.leaf_indices)
        num_permutations = 1000
        max_examples = 32000

        if self.depth > 3:
            per_depth_num_permutations = min(num_permutations, math.factorial(num_leaves), max_examples // num_leaves)
            permutations = [np.random.permutation(range(1, num_leaves + 1)) for _ in
                            range(per_depth_num_permutations)]
        else:
            permutations = random.sample(list(itertools.permutations(range(1, num_leaves + 1))),
                                         min(num_permutations, math.factorial(num_leaves)))

        return itertools.chain.from_iterable(

            zip(range(1, num_leaves + 1), itertools.repeat(perm))
            for perm in permutations)

    def get_nodes_features(self, combination):
        # combination: a list of indices
        # Each leaf contains a one-hot encoding of a key, and a one-hot encoding of the value
        # Every other node is empty, for now
        selected_key, values = combination

        # The root is [one-hot selected key] + [0 ... 0]
        nodes = [ (selected_key, 0) ]

        for i in range(1, self.num_nodes):
            if i in self.leaf_indices:
                leaf_num = self.leaf_indices.index(i)
                node = (leaf_num+1, values[leaf_num])
            else:
                node = (0, 0)
            nodes.append(node)
        return nodes

    def label(self, combination):
        selected_key, values = combination
        return int(values[selected_key - 1])

    def get_dims(self):
        # get input and output dims
        in_dim = len(self.leaf_indices)
        out_dim = len(self.leaf_indices)
        return in_dim, out_dim


class MyTreeDataset(InMemoryDataset):
    def __init__(self, root, train, seed, depth, transform=None, pre_transform=None, pre_filter=None):
        self.seed = seed
        self.depth = depth
        super().__init__(root, transform, pre_transform, pre_filter)
        idx = 0 if train else 1
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def process(self):
        # Read data into huge `Data` list.
        X_train, X_test, dim0, out_dim = DictionaryLookupDataset(self.depth, self.seed).generate_data(0.8)

        if self.pre_transform is not None:
            X_train = [self.pre_transform(data) for data in X_train]

        data, slices = self.collate(X_train)
        torch.save((data, slices), self.processed_paths[0])

        if self.pre_transform is not None:
            X_test = [self.pre_transform(data) for data in X_test]

        data, slices = self.collate(X_test)
        torch.save((data, slices), self.processed_paths[1])
