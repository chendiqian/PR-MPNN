import os
import pickle

import torch
from torch_geometric.data import InMemoryDataset

NAME = "GRAPHSAT"


class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, extra_path, transform=None, pre_transform=None, pre_filter=None):
        self.extra_path = extra_path
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [NAME+".pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed_' + self.extra_path)

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/"+NAME+".pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
