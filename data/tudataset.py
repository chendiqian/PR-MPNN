from typing import Callable, Optional

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.io import read_tu_data
from tqdm import tqdm


class MyTUDataset(TUDataset):
    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str, index: list = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        self.index = index
        super().__init__(root, name, transform, pre_transform, pre_filter, use_node_attr, use_edge_attr, cleaned)

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
        indices = self.index if self.index is not None else range(len(self))

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in indices]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in tqdm(data_list)]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self._data, self.slices, sizes), self.processed_paths[0])
