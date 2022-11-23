import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from ogb.io.read_graph_pyg import read_graph_pyg
from ogb.utils.url import decide_download, download_url, extract_zip
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.io import read_planetoid_data
from torch_geometric.utils import subgraph
from tqdm import tqdm

from subgraph.khop_subgraph import parallel_khop_neighbor, numba_k_hop_subgraph


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
                                      target_mask=target_mask,
                                      nnodes=torch.tensor([m.sum()])))

            pbar.close()
            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{sp}.pt'))


class MyPygNodePropPredDataset(InMemoryDataset):
    def __init__(self, name, root, training_split, khop, transform=None, pre_transform=None, meta_dict=None):
        """
        https://github.com/snap-stanford/ogb/blob/1c875697fdb20ab452b2c11cf8bfa2c0e88b5ad3/ogb/nodeproppred/dataset_pyg.py#L11
        """

        self.name = name  ## original name, e.g., ogbn-proteins
        self.khop = khop

        self.training_split = training_split
        assert training_split in ['train', 'valid', 'test']

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (
                not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name']  ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super().__init__(self.root, transform, pre_transform)

        path = osp.join(self.processed_dir, f'{training_split}.pt')
        self.data, self.slices = torch.load(path)

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        if self.is_hetero:
            raise NotImplementedError
        else:
            train_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)
            valid_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)
            test_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)

            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            if self.is_hetero:
                return ['edge_index_dict.npz']
            else:
                return ['data.npz']
        else:
            if self.is_hetero:
                return ['num-node-dict.csv.gz', 'triplet-type-list.csv.gz']
            else:
                file_names = ['edge']
                if self.meta_info['has_node_attr'] == 'True':
                    file_names.append('node-feat')
                if self.meta_info['has_edge_attr'] == 'True':
                    file_names.append('edge-feat')
                return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_dir(self):
        return osp.join(self.root, f'hop{self.khop}Inductive')

    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt', 'test.pt']

    def download(self):
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        if self.is_hetero:
            raise NotImplementedError
        else:
            data = \
                read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge,
                               additional_node_files=additional_node_files,
                               additional_edge_files=additional_edge_files, binary=self.binary)[0]

            ### adding prediction target
            if self.binary:
                node_label = np.load(osp.join(self.raw_dir, 'node-label.npz'))['node_label']
            else:
                node_label = pd.read_csv(osp.join(self.raw_dir, 'node-label.csv.gz'), compression='gzip',
                                         header=None).values

            if 'classification' in self.task_type:
                # detect if there is any nan
                if np.isnan(node_label).any():
                    data.y = torch.from_numpy(node_label).to(torch.float32)
                else:
                    data.y = torch.from_numpy(node_label).to(torch.long)

            else:
                data.y = torch.from_numpy(node_label).to(torch.float32)

        data = data if self.pre_transform is None else self.pre_transform(data)
        edge_index = data.edge_index.numpy()

        split = self.get_idx_split()  # dict[tensors]
        for k, v in split.items():
            split[k] = v.numpy()

        for sp in ['train', 'valid', 'test']:
            pbar = tqdm(split[sp])
            pbar.set_description(f'Processing {sp} dataset')
            data_list = []
            for node in pbar:
                np_node_idx, edge_index, _ = numba_k_hop_subgraph(edge_index,
                                                                  seed_node=node,
                                                                  khop=self.khop,
                                                                  num_nodes=data.num_nodes,
                                                                  relabel=True)

                target_mask = np_node_idx == node
                data_list.append(Data(x=data.x[np_node_idx],
                                      edge_index=torch.from_numpy(edge_index),
                                      y=data.y[np_node_idx][target_mask],
                                      target_mask=torch.from_numpy(target_mask),
                                      nnodes=torch.tensor([len(np_node_idx)])))

            pbar.close()
            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{sp}.pt'))
