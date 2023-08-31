import os.path as osp
import pickle
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from tqdm import tqdm


class PPGN_QM9(InMemoryDataset):
    def __init__(self, root, split, return_data=True, transform=None, pre_transform=None):
        self.split = split
        self.root = root
        assert split in ['train', 'valid', 'test']
        super().__init__(self.root, transform, pre_transform)

        self.slices = torch.load(osp.join(self.processed_dir, f'{split}_slice.pt'))
        self.data = torch.load(osp.join(self.processed_dir, f'{split}_data.pt')) if return_data else None

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return ['QM9_test.p', 'QM9_valid.p', 'QM9_train.p']

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt', f'{self.split}_slice.pt']

    # https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch/blob/master/utils/get_data.py
    def download(self):
        urls = [('https://www.dropbox.com/sh/acvh0sqgnvra53d/AAAxhVewejSl7gVMACa1tBUda/QM9_test.p?dl=1',
                'QM9_test.p'),
                ('https://www.dropbox.com/sh/acvh0sqgnvra53d/AAAOfEx-jGC6vvi43fh0tOq6a/QM9_val.p?dl=1',
                'QM9_valid.p'),
                ('https://www.dropbox.com/sh/acvh0sqgnvra53d/AADtx0EMRz5fhUNXaHFipkrza/QM9_train.p?dl=1',
                'QM9_train.p')]
        for url, filename in urls:
            _ = download_url(url, self.raw_dir)

    def process(self):
        data_list = []

        with open(osp.join(self.raw_dir, f'QM9_{self.split}.p'), 'rb') as f:
            data = pickle.load(f)

        for g in tqdm(data):
            pyg_data = Data(x=torch.from_numpy(g['usable_features']['x']),
                            y=torch.from_numpy(g['y']),
                            edge_index=torch.from_numpy(
                                g['original_features']['edge_index']),
                            edge_attr=torch.from_numpy(
                                g['original_features']['edge_attr']), )
            if self.pre_transform is not None:
                pyg_data = self.pre_transform(pyg_data)
            data_list.append(pyg_data)

        data, slices = self.collate(data_list)
        torch.save(data, osp.join(self.processed_dir, f'{self.split}_data.pt'))
        torch.save(slices, osp.join(self.processed_dir, f'{self.split}_slice.pt'))
