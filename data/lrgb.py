import os.path as osp
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.datasets.lrgb import LRGBDataset


# the only modification is renaming edge_label to y
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.LRGBDataset.html#torch_geometric.datasets.LRGBDataset
class MyLRGBDataset(LRGBDataset):
    def process_pcqm_contact(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pt'), 'rb') as f:
                graphs = torch.load(f)

            data_list = []
            for graph in tqdm(graphs, desc=f'Processing {split} dataset'):
                """
                PCQM-Contact
                Each `graph` is a tuple (x, edge_attr, edge_index,
                                        edge_label_index, edge_label)
                    Shape of x : [num_nodes, 9]
                    Shape of edge_attr : [num_edges, 3]
                    Shape of edge_index : [2, num_edges]
                    Shape of edge_label_index: [2, num_labeled_edges]
                    Shape of edge_label : [num_labeled_edges]

                    where,
                    num_labeled_edges are negative edges and link pred labels,
                    https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/loader/dataset/pcqm4mv2_contact.py#L192
                """
                x = graph[0]
                edge_attr = graph[1]
                edge_index = graph[2]
                edge_label_index = graph[3]
                edge_label = graph[4]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            edge_label_index=edge_label_index,
                            y=edge_label)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
