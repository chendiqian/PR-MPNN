import os
import csv
from argparse import Namespace
from functools import partial
from typing import Union, List, Optional

import numpy as np
import torch
from ml_collections import ConfigDict
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Subset
from torch_geometric.datasets import ZINC, WebKB, GNNBenchmarkDataset, LRGBDataset
from torch_geometric.transforms import Compose, AddRandomWalkPE, AddLaplacianEigenvectorPE
from torch_geometric.loader import DataLoader

from data.custom_datasets.qm9 import QM9
from data.custom_datasets.tree_dataset import MyTreeDataset, MyLeafColorDataset
from data.custom_datasets.planarsatpairsdataset import PlanarSATPairsDataset
from data.custom_datasets.symmetries import MySymDataset
from .const import DATASET_FEATURE_STAT_DICT
from .data_preprocess import (GraphExpandDim,
                              GraphToUndirected, GraphCoalesce,
                              GraphAttrToOneHot,
                              GraphAddRemainSelfLoop,
                              AugmentWithEdgeCandidate,
                              AugmentWithDumbAttr)

NUM_WORKERS = 0

DATASET = (PygGraphPropPredDataset,
           ZINC,
           MyTreeDataset,
           MyLeafColorDataset,
           MySymDataset,
           LRGBDataset,
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
    GraphAttrToOneHot: 0,  # low
    AugmentWithDumbAttr: 1,
    AugmentWithEdgeCandidate: 98,
    AddRandomWalkPE: 98,
    AddLaplacianEigenvectorPE: 98,
}


def get_additional_path(args: Union[Namespace, ConfigDict]):
    extra_path = ''
    heu = args.sample_configs.heuristic if hasattr(args.sample_configs, 'heuristic') else 'longest_path'
    directed = args.sample_configs.directed if hasattr(args.sample_configs, 'directed') else False
    extra_path += f'heu_{heu}_{"dir" if directed else "undir"}_{args.sample_configs.candid_pool}_'
    if hasattr(args.imle_configs, 'rwse') or hasattr(args, 'rwse'):
        extra_path += 'rwse_'
    if hasattr(args.imle_configs, 'lap') or hasattr(args, 'lap'):
        extra_path += 'lap_'
    return extra_path if len(extra_path) else None


def get_pretransform(args: Union[Namespace, ConfigDict], pretransforms: Optional[List] = None):
    if pretransforms is None:
        pretransforms = []

    if hasattr(args, 'rwse'):
        pretransforms.append(AddRandomWalkPE(args.rwse.kernel, 'pestat_RWSE'))

    if hasattr(args, 'lap'):
        pretransforms.append(AddLaplacianEigenvectorPE(args.lap.max_freqs, 'EigVecs', is_undirected=True))

    # add edge candidates
    heu = args.sample_configs.heuristic if hasattr(args.sample_configs, 'heuristic') else 'longest_path'
    directed = args.sample_configs.directed if hasattr(args.sample_configs, 'directed') else False
    if hasattr(args.sample_configs, 'candid_pool') and args.sample_configs.candid_pool > 0:
        pretransforms.append(AugmentWithEdgeCandidate(heu, args.sample_configs.candid_pool, directed))
    # else normal gnn

    if pretransforms:
        pretransforms = sorted(pretransforms, key=lambda p: PRETRANSFORM_PRIORITY[type(p)], reverse=True)
        return Compose(pretransforms)
    else:
        return None


def get_data(args: Union[Namespace, ConfigDict], *_args):
    """
    Distributor function

    :param args:
    :return:
    """
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

    if 'ogbg' in args.dataset.lower():
        train_set, val_set, test_set, std = get_ogbg_data(args)
    elif args.dataset.lower() == 'zinc':
        train_set, val_set, test_set, std = get_zinc(args)
    elif args.dataset.lower().startswith('tree'):
        train_set, val_set, test_set, std = get_treedataset(args)
    elif args.dataset.lower().startswith('leafcolor'):
        train_set, val_set, test_set, std = get_leafcolordataset(args)
    elif args.dataset.lower().startswith('peptides'):
        train_set, val_set, test_set, std = get_peptides(args)
    elif args.dataset.lower().startswith('hetero'):
        train_set, val_set, test_set, std = get_heterophily(args)
    elif args.dataset.lower() == 'qm9':
        train_set, val_set, test_set, std = get_qm9(args)
    elif args.dataset.lower() in ['exp', 'cexp']:
        train_set, val_set, test_set, std = get_exp_dataset(args, 10)
    elif 'sym' in args.dataset.lower():
        train_set, val_set, test_set, std = get_sym_dataset(args)
    elif args.dataset.lower() == 'csl':
        train_set, val_set, test_set, std = get_CSL(args)
    else:
        raise ValueError

    dataloader = partial(DataLoader,
                         batch_size=args.batch_size,
                         shuffle=not args.debug,
                         num_workers=NUM_WORKERS)

    if isinstance(train_set, list):
        train_loaders = [dataloader(t) for t in train_set]
    elif isinstance(train_set, DATASET):
        train_loaders = [dataloader(train_set)]
    else:
        raise TypeError

    if isinstance(val_set, list):
        val_loaders = [dataloader(t) for t in val_set]
    elif isinstance(val_set, DATASET):
        val_loaders = [dataloader(val_set)]
    else:
        raise TypeError

    if isinstance(test_set, list):
        test_loaders = [dataloader(t) for t in test_set]
    elif isinstance(test_set, DATASET):
        test_loaders = [dataloader(test_set)]
    else:
        raise TypeError

    return train_loaders, val_loaders, test_loaders, std if std is not None else 1


def get_ogbg_data(args: Union[Namespace, ConfigDict]):
    pre_transform = get_pretransform(args, pretransforms=[GraphToUndirected()])

    # if there are specific pretransforms, create individual folders for the dataset
    datapath = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    dataset = PygGraphPropPredDataset(name=args.dataset,
                                      root=datapath,
                                      transform=None,
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
    pre_transform = get_pretransform(args, pretransforms=[
        GraphAddRemainSelfLoop(),
        GraphToUndirected(),
        GraphExpandDim(),
        GraphAttrToOneHot(DATASET_FEATURE_STAT_DICT['zinc']['node'],
                          DATASET_FEATURE_STAT_DICT['zinc']['edge'])])

    data_path = os.path.join(args.data_path, 'ZINC')
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = ZINC(data_path,
                     split='train',
                     subset=True,
                     transform=None,
                     pre_transform=pre_transform)

    val_set = ZINC(data_path,
                   split='val',
                   subset=True,
                   transform=None,
                   pre_transform=pre_transform)

    test_set = ZINC(data_path,
                    split='test',
                    subset=True,
                    transform=None,
                    pre_transform=pre_transform)

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    return train_set, val_set, test_set, None


def get_peptides(args: Union[Namespace, ConfigDict]):
    datapath = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)
    pre_transform = get_pretransform(args)

    train_set = LRGBDataset(root=datapath, name=args.dataset.lower(), split='train', pre_transform=pre_transform)
    val_set = LRGBDataset(root=datapath, name=args.dataset.lower(), split='val', pre_transform=pre_transform)
    test_set = LRGBDataset(root=datapath, name=args.dataset.lower(), split='test', pre_transform=pre_transform)

    if args.debug:
        train_set = train_set[:4]
        val_set = val_set[:4]
        test_set = test_set[:4]

    return train_set, val_set, test_set, None


def get_CSL(args):
    pre_transform = get_pretransform(args, pretransforms=[AugmentWithDumbAttr()])

    data_path = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    dataset = GNNBenchmarkDataset(data_path,
                                  name='CSL',
                                  transform=None,
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

    pre_transform = get_pretransform(args, pretransforms=[GraphCoalesce()])

    data_path = os.path.join(args.data_path, args.dataset)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = MyTreeDataset(data_path, True, 11, depth, transform=None, pre_transform=pre_transform)
    val_set = MyTreeDataset(data_path, False, 11, depth, transform=None, pre_transform=pre_transform)
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

    pre_transform = get_pretransform(args, pretransforms=[GraphCoalesce()])

    data_path = os.path.join(args.data_path, args.dataset)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = MyLeafColorDataset(data_path, True, 11, depth, transform=None, pre_transform=pre_transform)
    val_set = MyLeafColorDataset(data_path, False, 11, depth, transform=None, pre_transform=pre_transform)
    test_set = val_set

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    args['num_classes'] = max([s.y.item() for s in train_set]) + 1

    return train_set, val_set, test_set, None


def get_sym_dataset(args: Union[Namespace, ConfigDict]):

    pre_transform = get_pretransform(args)

    data_path = os.path.join(args.data_path, args.dataset)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = MySymDataset(data_path, 'train', 11, args.dataset.lower(), transform=None, pre_transform=pre_transform)
    val_set = MySymDataset(data_path, 'val', 11, args.dataset.lower(), transform=None, pre_transform=pre_transform)
    test_set = MySymDataset(data_path, 'test', 11, args.dataset.lower(), transform=None, pre_transform=pre_transform)

    return train_set, val_set, test_set, None


def get_heterophily(args):
    dataset_name = args.dataset.lower().split('_')[1]
    datapath = os.path.join(args.data_path, 'hetero_' + dataset_name)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    pre_transforms = get_pretransform(args, pretransforms=[GraphToUndirected()])

    splits = {'train': [], 'val': [], 'test': []}

    # they use split 0
    # https://github.com/luis-mueller/probing-graph-transformers/blob/1a86d6f48bc4f8c7189ba5d9fd72e4078110a641/graphgps/config/split_config.py#L16
    # https://github.com/luis-mueller/probing-graph-transformers/blob/main/graphgps/loader/split_generator.py#L48
    folds = [0]
    for split in ['train', 'val', 'test']:
        for fold in folds:
            dataset = WebKB(root=datapath,
                            name=dataset_name,
                            transform=None,
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
    pre_transform = get_pretransform(args, pretransforms=[GraphCoalesce()])

    data_path = os.path.join(args.data_path, 'QM9')
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    assert hasattr(args, 'task_id') and 0 <= args.task_id <= 12

    dataset_lists = dict()

    for split in ['train', 'valid', 'test']:

        dataset = QM9(data_path,
                      split=split,
                      transform=None,
                      pre_transform=pre_transform)

        dataset._data.y = dataset._data.y[:, args.task_id: args.task_id + 1]
        dataset_lists[split] = dataset

    train_set = dataset_lists['train']
    val_set = dataset_lists['valid']
    test_set = dataset_lists['test']

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

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

    return train_set, val_set, test_set, std[args.task_id]


def get_exp_dataset(args, num_fold=10):
    extra_path = get_additional_path(args)
    extra_path = extra_path if extra_path is not None else 'normal'
    pre_transform = get_pretransform(args, pretransforms=[GraphToUndirected(),
                                                          GraphExpandDim(),
                                                          GraphAttrToOneHot(
                                                                    DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                                                                    0)])

    dataset = PlanarSATPairsDataset(os.path.join(args.data_path, args.dataset.upper()),
                                    extra_path,
                                    transform=None,
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
