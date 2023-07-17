import gzip
import json
import os

import gdown
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm


def map_qm9_to_pyg(json_file, make_undirected=True, remove_dup=False):
    # We're making the graph undirected just like the original repo.
    # Note: make_undirected makes duplicate edges, so we need to preserve edge types.
    # Note: The original repo also add self-loops. We don't need that given how we see hops.
    edge_index = np.array([[g[0], g[2]] for g in json_file["graph"]]).T  # Edge Index
    edge_attributes = np.array([g[1] - 1 for g in json_file["graph"]]
    )  # Edge type (-1 to put in [0, 3] range)
    if make_undirected:  # This will invariably cost us edge types because we reduce duplicates
        edge_index_reverse = edge_index[[1, 0], :]
        # Concat and remove duplicates
        if remove_dup:
            edge_index = torch.LongTensor(
                np.unique(
                    np.concatenate([edge_index, edge_index_reverse], axis=1), axis=1
                )
            )
        else:
            edge_index = torch.LongTensor(
                np.concatenate([edge_index, edge_index_reverse], axis=1)
            )
            edge_attributes = torch.LongTensor(
                np.concatenate([edge_attributes, np.copy(edge_attributes)], axis=0)
            )
    x = torch.FloatTensor(np.array(json_file["node_features"]))
    y = torch.FloatTensor(np.array(json_file["targets"]).T)
    edge_attributes = torch.nn.functional.one_hot(edge_attributes, 4).to(torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attributes, y=y)


class QM9(InMemoryDataset):
    url = {
        'test': 'https://drive.google.com/uc?id=1FLaiGzYOCUcpRW9OjDleIBMShaeKB59D',
        'valid': 'https://drive.google.com/uc?id=1w_tlpI8ubspo4fOUXJkcLCY1oaTq01H8',
        'train': 'https://drive.google.com/uc?id=17O4AZXFND180UikOMZkCCCLiC5hJqjOo',
    }

    def __init__(self,
                 root,
                 split,
                 return_data=True,
                 transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'valid', 'test']
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.slices = torch.load(os.path.join(self.processed_dir, f'{split}_slice.pt'))
        self.data = torch.load(os.path.join(self.processed_dir, f'{split}_data.pt')) if return_data else None

    @property
    def raw_file_names(self):
        return ['train.jsonl.gz', 'train.jsonl.gz', 'train.jsonl.gz']

    @property
    def processed_file_names(self):
        return ['train_data.pt', 'valid_data.pt', 'test_data.pt',
                'train_slice.pt', 'valid_slice.pt', 'test_slice.pt']

    def download(self):
        gdown.download(url=self.url['train'], output=os.path.join(self.raw_dir, 'train.jsonl.gz'), quiet=False)
        gdown.download(url=self.url['test'], output=os.path.join(self.raw_dir, 'test.jsonl.gz'), quiet=False)
        gdown.download(url=self.url['valid'], output=os.path.join(self.raw_dir, 'valid.jsonl.gz'), quiet=False)

    def process(self):
        for split in ['train', 'valid', 'test']:
            print(f'processing {split} split')
            with gzip.open(os.path.join(self.raw_dir, split + '.jsonl.gz'), "r") as f:
                data = f.read().decode("utf-8")
                graphs = [json.loads(jline) for jline in data.splitlines()]

                pyg_graphs = []
                for graph in tqdm(graphs):
                    pyg_graph = map_qm9_to_pyg(graph, make_undirected=True, remove_dup=False)
                    if self.pre_transform is not None:
                        pyg_graph = self.pre_transform(pyg_graph)
                    pyg_graphs.append(pyg_graph)

                if self.pre_filter is not None:
                    pyg_graphs = [d for d in pyg_graphs if self.pre_filter(d)]

            d, s = self.collate(pyg_graphs)
            torch.save(d, os.path.join(self.processed_dir, f'{split}_data.pt'))
            torch.save(s, os.path.join(self.processed_dir, f'{split}_slice.pt'))
