# from https://github.com/KarolisMart/DropGNN/blob/main/gin-synthetic.py

# This implementation is based on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
# Datasets are implemented based on the description in the corresonding papers (see the paper for references)
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx

torch.set_printoptions(profile="full")
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import degree as pyg_degree


# Synthetic datasets

class SymmetrySet:
    def __init__(self):
        self.hidden_units = 0
        self.num_classes = 0
        self.num_features = 0
        self.num_nodes = 0

    def addports(self, data):
        data.ports = torch.zeros(data.num_edges, 1)
        degs = degree(data.edge_index[0], data.num_nodes, dtype=torch.long) # out degree of all nodes
        for n in range(data.num_nodes):
            deg = degs[n]
            ports = np.random.permutation(int(deg))
            for i, neighbor in enumerate(data.edge_index[1][data.edge_index[0]==n]):
                nb = int(neighbor)
                data.ports[torch.logical_and(data.edge_index[0]==n, data.edge_index[1]==nb), 0] = float(ports[i])
        return data

    def makefeatures(self, data):
        data.x = torch.ones((data.num_nodes, 1))
        data.id = torch.tensor(np.random.permutation(np.arange(data.num_nodes))).unsqueeze(1)
        return data

    def makedata(self):
        pass

class LimitsOne(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False

    def makedata(self):
        n_nodes = 16 # There are two connected components, each with 8 nodes
        
        ports = [1,1,2,2] * 8
        colors = [0, 1, 2, 3] * 4

        y = torch.tensor([0]* 8 + [1] * 8)
        edge_index = torch.tensor([[0,1,1,2, 2,3,3,0, 4,5,5,6, 6,7,7,4, 8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,8], [1,0,2,1, 3,2,0,3, 5,4,6,5, 7,6,4,7, 9,8,10,9,11,10,12,11,13,12,14,13,15,14,8,15]], dtype=torch.long)
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = torch.tensor(np.random.permutation(np.arange(n_nodes))).unsqueeze(1)
        data.ports = torch.tensor(ports).unsqueeze(1)
        return [data]

class LimitsTwo(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False

    def makedata(self):
        n_nodes = 16 # There are two connected components, each with 8 nodes

        ports = ([1,1,2,2,1,1,2,2] * 2 + [3,3,3,3]) * 2
        colors = [0, 1, 2, 3] * 4
        y = torch.tensor([0] * 8 + [1] * 8)
        edge_index = torch.tensor([[0,1,1,2,2,3,3,0, 4,5,5,6,6,7,7,4, 1,3,5,7, 8,9,9,10,10,11,11,8, 12,13,13,14,14,15,15,12, 9,15,11,13], [1,0,2,1,3,2,0,3, 5,4,6,5,7,6,4,7, 3,1,7,5, 9,8,10,9,11,10,8,11, 13,12,14,13,15,14,12,15, 15,9,13,11]], dtype=torch.long)
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = torch.tensor(np.random.permutation(np.arange(n_nodes))).unsqueeze(1)
        data.ports = torch.tensor(ports).unsqueeze(1)
        return [data]

class Triangles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 60
        self.graph_class = False

    def makedata(self):
        size = self.num_nodes
        generated = False
        while not generated:
            nx_g = nx.random_degree_sequence_graph([3] * size)
            data = from_networkx(nx_g)
            labels = [0] * size
            for n in range(size):
                for nb1 in data.edge_index[1][data.edge_index[0]==n]:
                    for nb2 in data.edge_index[1][data.edge_index[0]==n]:
                        if torch.logical_and(data.edge_index[0]==nb1, data.edge_index[1]==nb2).any():
                            labels[n] = 1
            generated = labels.count(0) >= 20 and labels.count(1) >= 20
        data.y = torch.tensor(labels)

        data = self.addports(data)
        data = self.makefeatures(data)
        return [data]

class LCC(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 3
        self.num_features = 1
        self.num_nodes = 10
        self.graph_class = False

    def makedata(self):
        generated = False
        while not generated:
            graphs = []
            labels = []
            i = 0
            while i < 6:
                size = 10
                nx_g = nx.random_degree_sequence_graph([3] * size)
                if nx.is_connected(nx_g):
                    i += 1
                    data = from_networkx(nx_g)
                    lbls = [0] * size
                    for n in range(size):
                        edges = 0
                        nbs = [int(nb) for nb in data.edge_index[1][data.edge_index[0]==n]]
                        for nb1 in nbs:
                            for nb2 in nbs:
                                if torch.logical_and(data.edge_index[0]==nb1, data.edge_index[1]==nb2).any():
                                    edges += 1
                        lbls[n] = int(edges/2)
                    data.y = torch.tensor(lbls)
                    labels.extend(lbls)
                    data = self.addports(data)
                    data = self.makefeatures(data)
                    graphs.append(data)
            generated = labels.count(0) >= 10 and labels.count(1) >= 10 and labels.count(2) >= 10 # Ensure the dataset is somewhat balanced

        return graphs

class FourCycles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.p = 4
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 4 * self.p
        self.graph_class = True

    def gen_graph(self, p):
        edge_index = None
        for i in range(p):
            e = torch.tensor([[i, p + i, 2 * p + i, 3 * p + i], [2 * p + i, 3 * p + i, i, p + i]], dtype=torch.long)
            if edge_index is None:
                edge_index = e
            else:
                edge_index = torch.cat([edge_index, e], dim=-1)
        top = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            top[i * p + t] = 1
        bottom = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            bottom[i * p + t] = 1
        for i, bit in enumerate(top):
            if bit:
                e = torch.tensor([[i // p, p + i % p], [p + i % p, i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        for i, bit in enumerate(bottom):
            if bit:
                e = torch.tensor([[2 * p + i // p, 3 * p + i % p], [3 * p + i % p, 2 * p + i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        return Data(edge_index=edge_index, num_nodes=self.num_nodes), any(np.logical_and(top, bottom))

    def makedata(self):
        size = 25
        p = self.p
        trues = []
        falses = []
        while len(trues) < size or len(falses) < size:
            data, label = self.gen_graph(p)
            data = self.makefeatures(data)
            data = self.addports(data)
            data.y = int(label)
            if label and len(trues) < size:
                trues.append(data)
            elif not label and len(falses) < size:
                falses.append(data)
        return trues + falses

class SkipCircles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 32
        self.num_classes = 10 # num skips
        self.num_features = 1
        self.num_nodes = 41
        self.graph_class = True
        self.makedata()

    def makedata(self):
        size=self.num_nodes
        skips = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]
        graphs = []
        for s, skip in enumerate(skips):
            edge_index = torch.tensor([[0, size-1], [size-1, 0]], dtype=torch.long)
            for i in range(size - 1):
                e = torch.tensor([[i, i+1], [i+1, i]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
            for i in range(size):
                e = torch.tensor([[i, i], [(i - skip) % size, (i + skip) % size]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
            data = Data(edge_index=edge_index, num_nodes=self.num_nodes)
            data = self.makefeatures(data)
            data = self.addports(data)

            # data.x = torch.cat([data.x, data.id.float()], dim=1)
            # data.x = torch.cat([data.x, torch.randint(0, 100, (data.x.size(0), 1), device=data.x.device) / 100.0], dim=1)
            data.edge_attr = data.ports.expand(-1, data.x.size(-1))

            data.y = torch.tensor(s)
            graphs.append(data)

        return graphs        

class MySymDataset(InMemoryDataset):
    def __init__(self, root, subset, seed, dataset, transform=None, pre_transform=None, pre_filter=None):
        self.seed = seed
        self.dataset_name = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        if subset == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif subset == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif subset == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])


    @property
    def processed_file_names(self):
        return [f'train_{self.dataset_name}.pt', f'val_{self.dataset_name}', f'test_{self.dataset_name}.pt']

    def process(self):
        if 'limits1' in self.dataset_name:
            dataset = LimitsOne()
        elif 'limits2' in self.dataset_name:
            dataset = LimitsTwo()
        elif '4cycles' in self.dataset_name:
            dataset = FourCycles()
        elif 'skipcircles' in self.dataset_name:
            dataset = SkipCircles()
        elif 'lcc' in self.dataset_name:
            dataset = LCC()
        elif 'triangles' in self.dataset_name:
            dataset = Triangles()
        else:
            raise ValueError(f"Unknown sym dataset: {self.dataset_name}")
        
        degs = []

        for g in dataset.makedata():
            deg = pyg_degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
            degs.append(deg.max())
        
        print(f'Mean Degree: {torch.stack(degs).float().mean()}')
        print(f'Max Degree: {torch.stack(degs).max()}')
        print(f'Min Degree: {torch.stack(degs).min()}')
        print(f'Number of graphs: {len(dataset.makedata())}')
        
        graph_classification = dataset.graph_class
        if graph_classification:
            print('Graph Clasification Task')
        else:
            print('Node Clasification Task')

        train_set = dataset.makedata()
        val_set = dataset.makedata()
        test_set = dataset.makedata()

        print('Datasets generated!')

        if self.pre_transform is not None:
            train_set = [self.pre_transform(data) for data in train_set]
            val_set = [self.pre_transform(data) for data in val_set]
            test_set = [self.pre_transform(data) for data in test_set]

        data_train, slices_train = self.collate(train_set)
        data_val, slices_val = self.collate(val_set)
        data_test, slices_test = self.collate(test_set)

        torch.save((data_train, slices_train), self.processed_paths[0])
        torch.save((data_val, slices_val), self.processed_paths[1])
        torch.save((data_test, slices_test), self.processed_paths[2])