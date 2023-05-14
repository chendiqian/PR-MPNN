import os
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import roc_auc_score
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.data import Data, download_url


class HeterophilicDataset(InMemoryDataset):
    def __init__(self, root, name, split, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.split = split
        super(HeterophilicDataset, self).__init__(
            root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.is_transductive = True

    @property
    def raw_file_names(self):
        return f'{self.name}.npz'

    @property
    def processed_file_names(self):
        return f'{self.name}_processed.pt'

    def download(self):
        root_url = 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/'

        datasets = {
            'actor': root_url + 'actor.npz',
            'amazon_ratings': root_url + 'amazon_ratings.npz',
            'chameleon': root_url + 'chameleon.npz',
            'chameleon_directed': root_url + 'chameleon_directed.npz',
            'chameleon_filtered': root_url + 'chameleon_filtered.npz',
            'chameleon_filtered_directed': root_url + 'chameleon_filtered_directed.npz',
            'cornell': root_url + 'cornell.npz',
            'minesweeper': root_url + 'minesweeper.npz',
            'questions': root_url + 'questions.npz',
            'roman_empire': root_url + 'roman_empire.npz',
            'squirrel': root_url + 'squirrel.npz',
            'squirrel_directed': root_url + 'squirrel_directed.npz',
            'squirrel_filtered': root_url + 'squirrel_filtered.npz',
            'squirrel_filtered_directed': root_url + 'squirrel_filtered_directed.npz',
            'texas': root_url + 'texas.npz',
            'texas_4_classes': root_url + 'texas_4_classes.npz',
            'tolokers': root_url + 'tolokers.npz',
            'wisconsin': root_url + 'wisconsin.npz'
        }

        if self.name not in datasets:
            raise ValueError(f'Unknown dataset: {self.name}')

        download_url(datasets[self.name], self.raw_dir)

    def process(self):
        print('Preparing data...')
        data = np.load(os.path.join(self.raw_dir, self.raw_file_names))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges']).t().contiguous()

        num_classes = len(labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets == 1:
            labels = labels.float()

        fold = 0  # placeholder for now

        train_masks = torch.tensor(data['train_masks'])[fold]
        val_masks = torch.tensor(data['val_masks'])[fold]
        test_masks = torch.tensor(data['test_masks'])[fold]

        self.num_data_splits = train_masks[0]

        train_idx_list = [torch.where(train_mask)[0]
                          for train_mask in train_masks]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

        if self.split == 'train':
            self.split_ids = train_idx_list[fold]
        elif self.split == 'val':
            self.split_ids = val_idx_list[fold]
        elif self.split == 'test':
            self.split_ids = test_idx_list[fold]

        data = Data(edge_index=edges, num_nodes=len(node_features), x=node_features,
                    y=labels, train_mask=train_masks, val_mask=val_masks, test_mask=test_masks)

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])