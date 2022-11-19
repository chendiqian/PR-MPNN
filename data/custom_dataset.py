import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.io import read_planetoid_data
from torch_geometric.utils import subgraph

from subgraph.khop_subgraph import parallel_khop_neighbor


class PlanetoidKhopInductive(InMemoryDataset):
    """
    Mainly adapts from
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/planetoid.html#Planetoid
    """
    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    geom_gcn_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                    'geom-gcn/master')

    def __init__(self, root: str, name: str, khop: int, training_split: str, split: str = "public",
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        self.khop = khop

        self.split = split.lower()
        self.training_split = training_split
        assert training_split in ['train', 'val', 'test']
        assert self.split in ['public', 'full', 'geom-gcn', 'random']

        super().__init__(root, transform, pre_transform)
        path = osp.join(self.processed_dir, f'{training_split}.pt')
        self.data, self.slices = torch.load(path)

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', 'raw')
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', f'hop{self.khop}Inductive')
        return osp.join(self.root, self.name, f'hop{self.khop}Inductive')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)
        if self.split == 'geom-gcn':
            for i in range(10):
                url = f'{self.geom_gcn_url}/splits/{self.name.lower()}'
                download_url(f'{url}_split_0.6_0.2_{i}.npz', self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)

        if self.split == 'geom-gcn':
            train_masks, val_masks, test_masks = [], [], []
            for i in range(10):
                name = f'{self.name.lower()}_split_0.6_0.2_{i}.npz'
                splits = np.load(osp.join(self.raw_dir, name))
                train_masks.append(torch.from_numpy(splits['train_mask']))
                val_masks.append(torch.from_numpy(splits['val_mask']))
                test_masks.append(torch.from_numpy(splits['test_mask']))
            data.train_mask = torch.stack(train_masks, dim=1)
            data.val_mask = torch.stack(val_masks, dim=1)
            data.test_mask = torch.stack(test_masks, dim=1)

        data = data if self.pre_transform is None else self.pre_transform(data)

        mask = parallel_khop_neighbor(data.edge_index.numpy(), data.num_nodes, self.khop)
        mask = torch.from_numpy(mask)

        indices = {'train': data.train_mask, 'val': data.val_mask, 'test': data.test_mask}

        for sp in ['train', 'val', 'test']:
            ind = indices[sp].nonzero().squeeze()
            pbar = tqdm(ind)
            pbar.set_description(f'Processing {sp} dataset')
            data_list = []
            for i in pbar:
                m = mask[i]
                target_mask = m.nonzero().squeeze() == i
                edge_index, _ = subgraph(subset=m,
                                         edge_index=data.edge_index,
                                         relabel_nodes=True,
                                         num_nodes=data.num_nodes)
                data_list.append(Data(x=data.x[m],
                                      edge_index=edge_index,
                                      y=data.y[m][target_mask],
                                      target_mask=target_mask))

            pbar.close()
            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{sp}.pt'))
