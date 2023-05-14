import os

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, download_url


class HeterophilicDataset(InMemoryDataset):
    def __init__(self, root, name, split, fold, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.split = split
        super(HeterophilicDataset, self).__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(os.path.join(self.processed_dir, split + str(fold) + '.pt'))

    @property
    def raw_file_names(self):
        return f'{self.name}.npz'

    @property
    def processed_file_names(self):
        return [split + str(fold) + '.pt' for split in ['train', 'val', 'test'] for fold in range(10)]

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

        for split in ['train', 'val', 'test']:
            for fold in range(10):
                print(f'processing {split} split {fold}th fold')
                mask = torch.tensor(data[f'{split}_masks'][fold])
                graph = Data(edge_index=edges,
                            num_nodes=len(node_features),
                            x=node_features,
                            y=labels[mask],
                            transductive_mask=mask)

                graph = graph if self.pre_transform is None else self.pre_transform(graph)
                torch.save(self.collate([graph]), os.path.join(self.processed_dir, split + str(fold) + '.pt'))
