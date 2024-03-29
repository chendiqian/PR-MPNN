import os
import csv
from argparse import Namespace
from collections import defaultdict
from functools import partial
from typing import Union, List, Optional

import numpy as np
import torch
from ml_collections import ConfigDict
from networkx import kamada_kawai_layout
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Subset, DataLoader as PTDataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC, TUDataset, WebKB, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import Compose, AddRandomWalkPE, AddLaplacianEigenvectorPE
from torch_geometric.utils import degree as pyg_degree

from data.custom_datasets.peptides_func import PeptidesFunctionalDataset
from data.custom_datasets.peptides_struct import PeptidesStructuralDataset
from data.custom_datasets.qm9 import QM9
from data.custom_datasets.tree_dataset import MyTreeDataset, MyLeafColorDataset
from data.custom_datasets.tudataset import MyTUDataset
from data.custom_datasets.planarsatpairsdataset import PlanarSATPairsDataset
from data.custom_datasets.symmetries import MySymDataset
from data.utils.datatype_utils import AttributedDataLoader
from data.utils.plot_utils import circular_tree_layout
from .const import DATASET_FEATURE_STAT_DICT
from .data_preprocess import (GraphExpandDim,
                              GraphToUndirected, GraphCoalesce,
                              AugmentwithNumbers,
                              GraphAttrToOneHot,
                              GraphAddRemainSelfLoop,
                              AugmentWithEdgeCandidate,
                              AugmentWithPlotCoordinates,
                              AugmentWithDumbAttr,
                              DropEdge,
                              collate_fn_with_origin_list)
from .random_baseline import AugmentWithRandomRewiredGraphs, collate_random_rewired_batch

NUM_WORKERS = 0

DATASET = (PygGraphPropPredDataset,
           ZINC,
           MyTreeDataset,
           MyLeafColorDataset,
           MySymDataset,
           MyTUDataset,
           TUDataset,
           PeptidesStructuralDataset,
           PeptidesFunctionalDataset,
           WebKB,
           PlanarSATPairsDataset,
           QM9,
           Subset, GNNBenchmarkDataset)

# sort keys, some pre_transform should be executed first
PRETRANSFORM_PRIORITY = {
    GraphExpandDim: 0,  # low
    GraphAddRemainSelfLoop: 100,  # highest
    GraphToUndirected: 99,  # high
    GraphCoalesce: 99,
    AugmentwithNumbers: 0,  # low
    GraphAttrToOneHot: 0,  # low
    AugmentWithDumbAttr: 1,
    AugmentWithEdgeCandidate: 98,
    AddRandomWalkPE: 98,
    AddLaplacianEigenvectorPE: 98,
    AugmentWithPlotCoordinates: 98,
}


def get_additional_path(args: Union[Namespace, ConfigDict]):
    extra_path = ''
    if args.sample_configs.sample_policy == 'edge_candid':
        heu = args.sample_configs.heuristic if hasattr(args.sample_configs, 'heuristic') else 'longest_path'
        directed = args.sample_configs.directed if hasattr(args.sample_configs, 'directed') else False
        extra_path += f'EdgeCandidates_{heu}_{"dir" if directed else "undir"}_{args.sample_configs.candid_pool}_'
    if hasattr(args.imle_configs, 'rwse') or hasattr(args, 'rwse'):
        extra_path += 'rwse_'
    if hasattr(args.imle_configs, 'lap') or hasattr(args, 'lap'):
        extra_path += 'lap_'
    return extra_path if len(extra_path) else None


def get_transform(args: Union[Namespace, ConfigDict]):
    transform = []

    # robustness ablation
    assert not (hasattr(args, 'perturb_add') and hasattr(args, 'perturb_del'))
    if hasattr(args, 'perturb_del'):
        directed = args.sample_configs.directed if hasattr(args, 'sample_configs') else False
        transform.append(DropEdge(args.perturb_del, directed))
    if hasattr(args, 'perturb_add'):
        directed = args.sample_configs.directed if hasattr(args, 'sample_configs') else False
        transform.append(DropEdge(args.perturb_add, directed))

    # uniform random baseline
    if args.sample_configs.sample_policy == 'add_del':
        transform.append(AugmentWithRandomRewiredGraphs(sample_k_add=args.sample_configs.sample_k,
                                                   sample_k_del=args.sample_configs.sample_k2,
                                                   include_original_graph=args.sample_configs.include_original_graph,
                                                   in_place=args.sample_configs.in_place,
                                                   ensemble=args.sample_configs.ensemble,
                                                   separate=args.sample_configs.separate,
                                                   directed=args.sample_configs.directed,
                                                   ))

    if transform:
        return Compose(transform)
    else:
        return None


def get_pretransform(args: Union[Namespace, ConfigDict], extra_pretransforms: Optional[List] = None):
    pretransform = [AugmentwithNumbers()]
    if extra_pretransforms is not None:
        pretransform = pretransform + extra_pretransforms

    if hasattr(args.imle_configs, 'rwse'):
        pretransform.append(AddRandomWalkPE(args.imle_configs.rwse.kernel, 'pestat_RWSE'))
    elif hasattr(args, 'rwse'):
        pretransform.append(AddRandomWalkPE(args.rwse.kernel, 'pestat_RWSE'))

    if hasattr(args.imle_configs, 'lap'):
        pretransform.append(AddLaplacianEigenvectorPE(args.imle_configs.lap.max_freqs, 'EigVecs', is_undirected=True))
    elif hasattr(args, 'lap'):
        pretransform.append(AddLaplacianEigenvectorPE(args.lap.max_freqs, 'EigVecs', is_undirected=True))

    # add edge candidates or bidirectional
    if args.sample_configs.sample_policy == 'edge_candid':
        heu = args.sample_configs.heuristic if hasattr(args.sample_configs, 'heuristic') else 'longest_path'
        directed = args.sample_configs.directed if hasattr(args.sample_configs, 'directed') else False
        pretransform.append(AugmentWithEdgeCandidate(heu, args.sample_configs.candid_pool, directed))

    pretransform = sorted(pretransform, key=lambda p: PRETRANSFORM_PRIORITY[type(p)], reverse=True)
    return Compose(pretransform)

def get_data(args: Union[Namespace, ConfigDict], *_args):
    """
    Distributor function

    :param args:
    :return:
    """
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

    task = 'graph'
    qm9_task_id = None
    separate_std = 'qm9' in args.dataset.lower()
    if 'ogbg' in args.dataset.lower():
        train_set, val_set, test_set, std = get_ogbg_data(args)
    elif args.dataset.lower() == 'zinc':
        train_set, val_set, test_set, std = get_zinc(args)
    elif args.dataset.lower() == 'alchemy':
        train_set, val_set, test_set, std = get_alchemy(args)
    elif args.dataset.lower() == 'proteins':
        train_set, val_set, test_set, std = get_TU(args, name='PROTEINS_full')
    elif args.dataset.lower() == 'mutag':
        train_set, val_set, test_set, std = get_TU(args, name='MUTAG')
    elif args.dataset.lower() == 'ptc_mr':
        train_set, val_set, test_set, std = get_TU(args, name='PTC_MR')
    elif args.dataset.lower() == 'nci1':
        train_set, val_set, test_set, std = get_TU(args, name='NCI1')
    elif args.dataset.lower() == 'nci109':
        train_set, val_set, test_set, std = get_TU(args, name='NCI109')
    elif args.dataset.lower() == 'imdb-m':
        train_set, val_set, test_set, std = get_imdbm(args)
    elif args.dataset.lower() == 'imdb-b':
        train_set, val_set, test_set, std = get_imdbb(args)
    elif args.dataset.lower().startswith('tree'):
        train_set, val_set, test_set, std = get_treedataset(args)
    elif args.dataset.lower().startswith('leafcolor'):
        train_set, val_set, test_set, std = get_leafcolordataset(args)
    elif args.dataset.lower().startswith('peptides-struct'):
        train_set, val_set, test_set, std = get_peptides(args, set='struct')
    elif args.dataset.lower().startswith('hetero'):
        train_set, val_set, test_set, std = get_heterophily(args)
        task = 'node'
    elif args.dataset.lower().startswith('peptides-func'):
        train_set, val_set, test_set, std = get_peptides(args, set='func')
    elif args.dataset.lower() == 'qm9':
        train_set, val_set, test_set, std, qm9_task_id = get_qm9(args)
    elif args.dataset.lower() in ['exp', 'cexp']:
        train_set, val_set, test_set, std = get_exp_dataset(args, 10)
    elif 'sym' in args.dataset.lower():
        train_set, val_set, test_set, std = get_sym_dataset(args)
    elif args.dataset.lower() == 'csl':
        train_set, val_set, test_set, std = get_CSL(args)
    else:
        raise ValueError

    # calculate degree stats for PNA model
    if 'pna' in args.model.lower():
        target_dataset = train_set[0] if isinstance(train_set, list) else train_set
        degs = torch.cat([pyg_degree(g.edge_index[1],
                                     num_nodes=g.num_nodes,
                                     dtype=torch.long) for g in target_dataset], dim=0)
        args.ds_deg = torch.bincount(degs)

    if args.imle_configs is not None:
        # collator that returns a batch and a list
        dataloader = partial(PTDataLoader,
                             batch_size=args.batch_size,
                             shuffle=not args.debug,
                             num_workers=NUM_WORKERS,
                             collate_fn=collate_fn_with_origin_list)
    elif args.sample_configs.sample_policy is not None:
        # collator for sampled graphs
        dataloader = partial(PTDataLoader,
                             batch_size=args.batch_size,
                             shuffle=not args.debug,
                             num_workers=NUM_WORKERS,
                             collate_fn=partial(collate_random_rewired_batch,
                                                include_org=args.sample_configs.include_original_graph)
                             )
    else:
        # PyG removes the collate function passed in
        dataloader = partial(PyGDataLoader,
                             batch_size=args.batch_size,
                             shuffle=not args.debug,
                             num_workers=NUM_WORKERS)

    if isinstance(train_set, list):
        train_loaders = [AttributedDataLoader(
            loader=dataloader(t),
            std=std if not separate_std else std[i],
            task=task) for i, t in enumerate(train_set)]
    elif isinstance(train_set, DATASET):
        train_loaders = [AttributedDataLoader(
            loader=dataloader(train_set),
            std=std,
            task=task)]
    else:
        raise TypeError

    if isinstance(val_set, list):
        val_loaders = [AttributedDataLoader(
            loader=dataloader(t),
            std=std if not separate_std else std[i],
            task=task) for i, t in enumerate(val_set)]
    elif isinstance(val_set, DATASET):
        val_loaders = [AttributedDataLoader(
            loader=dataloader(val_set),
            std=std,
            task=task)]
    else:
        raise TypeError

    if isinstance(test_set, list):
        test_loaders = [AttributedDataLoader(
            loader=dataloader(t),
            std=std if not separate_std else std[i],
            task=task) for i, t in enumerate(test_set)]
    elif isinstance(test_set, DATASET):
        test_loaders = [AttributedDataLoader(
            loader=dataloader(test_set),
            std=std,
            task=task)]
    else:
        raise TypeError

    return train_loaders, val_loaders, test_loaders, qm9_task_id


def get_ogbg_data(args: Union[Namespace, ConfigDict]):
    pre_transform = get_pretransform(args, extra_pretransforms=[
        # AugmentWithPlotCoordinates(layout=kamada_kawai_layout),
        GraphToUndirected()])
    transform = get_transform(args)

    # if there are specific pretransforms, create individual folders for the dataset
    datapath = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    dataset = PygGraphPropPredDataset(name=args.dataset,
                                      root=datapath,
                                      transform=transform,
                                      pre_transform=pre_transform)
    dataset.data.y = dataset.data.y.float()
    split_idx = dataset.get_idx_split()

    train_idx = split_idx["train"] if not args.debug else split_idx["train"][:16]
    val_idx = split_idx["valid"] if not args.debug else split_idx["valid"][:16]
    test_idx = split_idx["test"] if not args.debug else split_idx["test"][:16]

    train_set = dataset[train_idx]
    val_set = dataset[val_idx]
    test_set = dataset[test_idx]

    return train_set, val_set, test_set, None


def get_zinc(args: Union[Namespace, ConfigDict]):
    pre_transform = get_pretransform(args, extra_pretransforms=[
        AugmentWithPlotCoordinates(layout=kamada_kawai_layout),
        GraphAddRemainSelfLoop(),
        GraphToUndirected(),
        GraphExpandDim(),
        GraphAttrToOneHot(DATASET_FEATURE_STAT_DICT['zinc']['node'],
                          DATASET_FEATURE_STAT_DICT['zinc']['edge'])])
    transform = get_transform(args)

    data_path = os.path.join(args.data_path, 'ZINC')
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = ZINC(data_path,
                     split='train',
                     subset=True,
                     transform=transform,
                     pre_transform=pre_transform)

    val_set = ZINC(data_path,
                   split='val',
                   subset=True,
                   transform=transform,
                   pre_transform=pre_transform)

    test_set = ZINC(data_path,
                    split='test',
                    subset=True,
                    transform=transform,
                    pre_transform=pre_transform)

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    return train_set, val_set, test_set, None

def get_peptides(args: Union[Namespace, ConfigDict], set='struct'):
    datapath = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)
    pre_transform = get_pretransform(args, extra_pretransforms=None)
    transform = get_transform(args)

    if set == 'struct':
        dataset = PeptidesStructuralDataset(root=datapath, transform=transform, pre_transform=pre_transform)
    elif set == 'func':
        dataset = PeptidesFunctionalDataset(root=datapath, transform=transform, pre_transform=pre_transform)
    else:
        raise ValueError(f"Unknown peptides set: {set}")

    split_idx = dataset.get_idx_split()
    
    train_set, val_set, test_set = dataset[split_idx['train']], dataset[split_idx['val']], dataset[split_idx['test']]

    if args.debug:
        if hasattr(args, 'debug_samples'):
            debug_samples = args.debug_samples
        else:
            debug_samples = 16
        train_set = train_set[:debug_samples]
        val_set = val_set[:debug_samples]
        test_set = test_set[:debug_samples]

    return train_set, val_set, test_set, None


def get_alchemy(args: Union[Namespace, ConfigDict]):
    pre_transform = get_pretransform(args, extra_pretransforms=[AugmentWithPlotCoordinates(layout=kamada_kawai_layout)])
    transform = get_transform(args)

    data_path = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    infile = open("datasets/indices/train_al_10.index", "r")
    for line in infile:
        indices_train = line.split(",")
        indices_train = [int(i) for i in indices_train]

    infile = open("datasets/indices/val_al_10.index", "r")
    for line in infile:
        indices_val = line.split(",")
        indices_val = [int(i) for i in indices_val]

    infile = open("datasets/indices/test_al_10.index", "r")
    for line in infile:
        indices_test = line.split(",")
        indices_test = [int(i) for i in indices_test]

    dataset = MyTUDataset(data_path,
                          name="alchemy_full",
                          index=indices_train + indices_val + indices_test,
                          transform=transform,
                          pre_transform=pre_transform)

    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std

    train_set = dataset[:len(indices_train)]
    val_set = dataset[len(indices_train): len(indices_train) + len(indices_val)]
    test_set = dataset[-len(indices_test):]

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    return train_set, val_set, test_set, std


def get_TU(args, name='PROTEINS_full'):
    pre_transform = get_pretransform(args, extra_pretransforms=[
        AugmentWithPlotCoordinates(layout=kamada_kawai_layout),
        GraphToUndirected(),
        GraphExpandDim()])
    transform = get_transform(args)

    data_path = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    dataset = TUDataset(data_path,
                        name=name,
                        transform=transform,
                        pre_transform=pre_transform)

    train_splits = []
    val_splits = []

    labels = dataset.data.y.tolist()
    dataset.data.y = dataset.data.y.float()
    
    for fold_idx in range(0, 10):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=fold_idx)


        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        train_splits.append(Subset(dataset, train_idx))
        val_splits.append(Subset(dataset, test_idx))

    ## one split
    # num_training = int(len(dataset) * 0.8)
    # num_val = int(len(dataset) * 0.1)
    # num_test = len(dataset) - num_val - num_training
    # train_set, val_set, test_set = random_split(dataset,
    #                                             [num_training, num_val, num_test],
    #                                             generator=torch.Generator().manual_seed(0))

    if args.debug:
        train_splits = train_splits[0]
        val_splits = val_splits[0]

    test_splits = val_splits

    return train_splits, val_splits, test_splits, None


def get_imdbm(args):
    pre_transform = get_pretransform(args, extra_pretransforms=[
        AugmentWithPlotCoordinates(layout=kamada_kawai_layout),
        AugmentWithDumbAttr()])
    transform = get_transform(args)

    data_path = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    dataset = TUDataset(data_path,
                        name='IMDB-MULTI',
                        transform=transform,
                        pre_transform=pre_transform)

    train_splits = []
    val_splits = []

    labels = dataset.data.y.tolist()

    for fold_idx in range(0, 10):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=fold_idx)

        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        train_splits.append(Subset(dataset, train_idx))
        val_splits.append(Subset(dataset, test_idx))

    ## one split
    # num_training = int(len(dataset) * 0.8)
    # num_val = int(len(dataset) * 0.1)
    # num_test = len(dataset) - num_val - num_training
    # train_set, val_set, test_set = random_split(dataset,
    #                                             [num_training, num_val, num_test],
    #                                             generator=torch.Generator().manual_seed(0))

    if args.debug:
        train_splits = train_splits[0]
        val_splits = val_splits[0]

    test_splits = val_splits

    return train_splits, val_splits, test_splits, None


def get_imdbb(args):
    pre_transform = get_pretransform(args, extra_pretransforms=[
        AugmentWithPlotCoordinates(layout=kamada_kawai_layout),
        AugmentWithDumbAttr(),
        GraphExpandDim()])
    transform = get_transform(args)

    data_path = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    dataset = TUDataset(data_path,
                        name='IMDB-BINARY',
                        transform=transform,
                        pre_transform=pre_transform)

    train_splits = []
    val_splits = []

    labels = dataset.data.y.tolist()
    dataset.data.y = dataset.data.y.float()

    for fold_idx in range(0, 10):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=fold_idx)

        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        train_splits.append(Subset(dataset, train_idx))
        val_splits.append(Subset(dataset, test_idx))

    ## one split
    # num_training = int(len(dataset) * 0.8)
    # num_val = int(len(dataset) * 0.1)
    # num_test = len(dataset) - num_val - num_training
    # train_set, val_set, test_set = random_split(dataset,
    #                                             [num_training, num_val, num_test],
    #                                             generator=torch.Generator().manual_seed(0))

    if args.debug:
        train_splits = train_splits[0]
        val_splits = val_splits[0]

    test_splits = val_splits

    return train_splits, val_splits, test_splits, None


def get_CSL(args):
    pre_transform = get_pretransform(args, extra_pretransforms=[
        AugmentWithPlotCoordinates(layout=kamada_kawai_layout),
        AugmentWithDumbAttr(),
    ])
    transform = get_transform(args)

    data_path = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    dataset = GNNBenchmarkDataset(data_path,
                                  name='CSL',
                                  transform=transform,
                                  pre_transform=pre_transform)

    def get_all_split_idx(dataset):
        """
            - Split total number of graphs into 3 (train, val and test) in 3:1:1
            - Stratified split proportionate to original distribution of data with respect to classes
            - Using sklearn to perform the split and then save the indexes
            - Preparing 5 such combinations of indexes split to be used in Graph NNs
            - As with KFold, each of the 5 fold have unique test set.
        """
        root_idx_dir = f'{data_path}/CSL/splits/'
        if not os.path.exists(root_idx_dir):
            os.makedirs(root_idx_dir)
        all_idx = {}

        # If there are no idx files, do the split and store the files
        if not (os.path.exists(root_idx_dir + dataset.name + '_train.index')):
            print("[!] Splitting the data into train/val/test ...")

            # Using 5-fold cross val as used in RP-GNN paper
            k_splits = 5

            cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
            labels = dataset.data.y.squeeze().numpy()

            for indexes in cross_val_fold.split(labels, labels):
                remain_index, test_index = indexes[0], indexes[1]

                # Gets final 'train' and 'val'
                train, val, _, __ = train_test_split(remain_index,
                                                     range(len(remain_index)),
                                                     test_size=0.25,
                                                     stratify=labels[remain_index])

                with open(root_idx_dir + dataset.name + '_train.index', 'a+') as f:
                    f_train_w = csv.writer(f)
                    f_train_w.writerow(train)
                with open(root_idx_dir + dataset.name + '_val.index', 'a+') as f:
                    f_val_w = csv.writer(f)
                    f_val_w.writerow(val)
                with open(root_idx_dir + dataset.name + '_test.index', 'a+') as f:
                    f_test_w = csv.writer(f)
                    f_test_w.writerow(test_index)

            print("[!] Splitting done!")

        # reading idx from the files
        for section in ['train', 'val', 'test']:
            with open(root_idx_dir + dataset.name + '_' + section + '.index', 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        return all_idx

    splits = get_all_split_idx(dataset)

    train_splits = [Subset(dataset, splits['train'][i]) for i in range(5)]
    val_splits = [Subset(dataset, splits['val'][i]) for i in range(5)]
    test_splits = [Subset(dataset, splits['test'][i]) for i in range(5)]

    if args.debug:
        train_splits = train_splits[0]
        val_splits = val_splits[0]
        test_splits = test_splits[0]

    return train_splits, val_splits, test_splits, None


def get_treedataset(args: Union[Namespace, ConfigDict]):
    depth = int(args.dataset.lower().split('_')[1])
    assert 2 <= depth <= 8

    pre_transform = get_pretransform(args, extra_pretransforms=[GraphCoalesce(), AugmentWithPlotCoordinates(layout=circular_tree_layout)])
    # pre_transform = get_pretransform(args, extra_pretransforms=[GraphCoalesce(), GraphRedirect(depth)])
    transform = get_transform(args)

    data_path = os.path.join(args.data_path, args.dataset)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = MyTreeDataset(data_path, True, 11, depth, transform=transform, pre_transform=pre_transform)
    val_set = MyTreeDataset(data_path, False, 11, depth, transform=transform, pre_transform=pre_transform)
    # min is 1
    train_set.data.y -= 1
    val_set.data.y -= 1
    test_set = val_set

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    return train_set, val_set, test_set, None

def get_leafcolordataset(args: Union[Namespace, ConfigDict]):
    depth = int(args.dataset.lower().split('_')[1])
    assert 2 <= depth <= 8

    pre_transform = get_pretransform(args, extra_pretransforms=[GraphCoalesce(), AugmentWithPlotCoordinates(layout=circular_tree_layout)])
    transform = get_transform(args)

    data_path = os.path.join(args.data_path, args.dataset)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = MyLeafColorDataset(data_path, True, 11, depth, transform=transform, pre_transform=pre_transform)
    val_set = MyLeafColorDataset(data_path, False, 11, depth, transform=transform, pre_transform=pre_transform)
    test_set = val_set

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    args['num_classes'] = max([s.y.item() for s in train_set]) + 1

    return train_set, val_set, test_set, None



def get_sym_dataset(args: Union[Namespace, ConfigDict]):

    pre_transform = get_pretransform(args, extra_pretransforms=[])
    transform = get_transform(args)

    data_path = os.path.join(args.data_path, args.dataset)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = MySymDataset(data_path, 'train', 11, args.dataset.lower(), transform=transform, pre_transform=pre_transform)
    val_set = MySymDataset(data_path, 'val', 11, args.dataset.lower(), transform=transform, pre_transform=pre_transform)
    test_set = MySymDataset(data_path, 'test', 11, args.dataset.lower(), transform=transform, pre_transform=pre_transform)
    
    return train_set, val_set, test_set, None


def get_heterophily(args):
    dataset_name = args.dataset.lower().split('_')[1]
    datapath = os.path.join(args.data_path, 'hetero_' + dataset_name)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    pre_transforms = get_pretransform(args, extra_pretransforms=[GraphToUndirected()])
    transform = get_transform(args)

    splits = {'train': [], 'val': [], 'test': []}

    # they use split 0
    # https://github.com/luis-mueller/probing-graph-transformers/blob/1a86d6f48bc4f8c7189ba5d9fd72e4078110a641/graphgps/config/split_config.py#L16
    # https://github.com/luis-mueller/probing-graph-transformers/blob/main/graphgps/loader/split_generator.py#L48
    folds = [0]
    for split in ['train', 'val', 'test']:
        for fold in folds:
            dataset = WebKB(root=datapath,
                            name=dataset_name,
                            transform=transform,
                            pre_transform=pre_transforms)
            mask = getattr(dataset.data, f'{split}_mask')
            mask = mask[:, fold]
            dataset.data.y = dataset.data.y[mask]
            dataset.data.transductive_mask = mask

            splits[split].append(dataset)

    train_set, val_set, test_set = splits['train'], splits['val'], splits['test']
    if args.debug:
        train_set = train_set[0]
        val_set = val_set[0]
        test_set = test_set[0]

    return train_set, val_set, test_set, None


def get_qm9(args: Union[Namespace, ConfigDict]):
    pre_transform = get_pretransform(args, extra_pretransforms=[
        AugmentWithPlotCoordinates(layout=kamada_kawai_layout),
        GraphCoalesce()])
    transform = get_transform(args)

    data_path = os.path.join(args.data_path, 'QM9')
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    if hasattr(args, 'task_id'):
        if isinstance(args.task_id, int):
            assert 0 <= args.task_id <= 12
            task_id = [args.task_id]
        else:
            raise TypeError
    else:
        task_id = list(range(13))

    dataset_lists = defaultdict(list)

    for split in ['train', 'valid', 'test']:

        dataset = QM9(data_path,
                      split=split,
                      transform=transform,
                      pre_transform=pre_transform)

        for i in task_id:
            new_data = Data()
            for k, v in dataset._data._store.items():
                if k != 'y':
                    setattr(new_data, k, v)
                else:
                    setattr(new_data, k, v[:, i:i + 1])

            d = QM9(data_path,
                    split=split,
                    return_data=False,
                    transform=transform,
                    pre_transform=pre_transform)
            d.data = new_data
            dataset_lists[split].append(d)

    train_set = dataset_lists['train']
    val_set = dataset_lists['valid']
    test_set = dataset_lists['test']

    if args.debug:
        train_set = [t[:16] for t in train_set]
        val_set = [t[:16] for t in val_set]
        test_set = [t[:16] for t in test_set]

    # https://github.com/radoslav11/SP-MPNN/blob/main/src/experiments/run_gr.py#L22
    norm_const = [
        0.066513725,
        0.012235489,
        0.071939046,
        0.033730778,
        0.033486113,
        0.004278493,
        0.001330901,
        0.004165489,
        0.004128926,
        0.00409976,
        0.004527465,
        0.012292586,
        0.037467458,
    ]
    std = 1. / torch.tensor(norm_const, dtype=torch.float)

    return train_set, val_set, test_set, [std[i] for i in task_id], task_id


def get_exp_dataset(args, num_fold=10):
    extra_path = get_additional_path(args)
    extra_path = extra_path if extra_path is not None else 'normal'
    pre_transform = get_pretransform(args, extra_pretransforms=[GraphToUndirected(),
                                                                GraphExpandDim(),
                                                                AugmentWithPlotCoordinates(
                                                                    layout=kamada_kawai_layout),
                                                                GraphAttrToOneHot(
                                                                    DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                                                                    0)])
    transform = get_transform(args)

    dataset = PlanarSATPairsDataset(os.path.join(args.data_path, args.dataset.upper()),
                                    extra_path,
                                    transform=transform,
                                    pre_transform=pre_transform)
    dataset.data.y = dataset.data.y.float()

    def separate_data(fold_idx, dataset):
        assert 0 <= fold_idx < 10, "fold_idx must be from 0 to 9."
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        labels = dataset.data.y.numpy()
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        return torch.tensor(train_idx), torch.tensor(test_idx), torch.tensor(test_idx)

    train_sets, val_sets, test_sets = [], [], []
    for idx in range(num_fold):
        train, val, test = separate_data(idx, dataset)
        train_set = dataset[train]
        val_set = dataset[val]
        test_set = dataset[test]

        train_sets.append(train_set)
        val_sets.append(val_set)
        test_sets.append(test_set)

    if args.debug:
        train_sets = train_sets[0]
        val_sets = val_sets[0]
        test_sets = test_sets[0]

    return train_sets, val_sets, test_sets, None
