import os
from argparse import Namespace
from typing import Tuple, Union, List, Optional

from ml_collections import ConfigDict
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from .const import DATASET_FEATURE_STAT_DICT, MAX_NUM_NODE_DICT
from .custom_dataset import PlanetoidKhopInductive, MyPygNodePropPredDataset, MyQM9
from .data_preprocess import (GraphExpandDim,
                              GraphToUndirected,
                              AugmentwithNNodes,
                              policy2transform,
                              GraphAttrToOneHot,
                              GraphAddRemainSelfLoop,
                              AugmentWithShortedPathDistance,
                              AugmentWithPPR,
                              AugmentWithRandomKNeighbors,
                              AugmentWithKhopMasks,
                              RandomSampleTopk
                              )
from .data_utils import AttributedDataLoader

DATASET = (PygGraphPropPredDataset, ZINC, PlanetoidKhopInductive, MyPygNodePropPredDataset)

NAME_DICT = {'zinc_full': "ZINC_full",
             'cora': 'Cora',
             'pubmed': 'PubMed'}

PRETRANSFORM_PRIORITY = {
    GraphExpandDim: 0,  # low
    GraphAddRemainSelfLoop: 100,  # highest
    GraphToUndirected: 99,  # high
    AugmentwithNNodes: 0,  # low
    GraphAttrToOneHot: 0,  # low
    AugmentWithShortedPathDistance: 98,
    AugmentWithPPR: 98,
    AugmentWithRandomKNeighbors: 0,
    AugmentWithKhopMasks: 0,
    RandomSampleTopk: 0,
}


def get_additional_path(args: Union[Namespace, ConfigDict]):
    extra_path = ''
    if hasattr(args.imle_configs, 'emb_spd') and args.imle_configs.emb_spd:
        extra_path += 'SPDaug_'
    if hasattr(args.imle_configs, 'emb_ppr') and args.imle_configs.emb_ppr:
        extra_path += 'PPRaug_'
    if args.sample_configs.sample_policy in ['khop']:
        extra_path += args.sample_configs.sample_policy + '_' + str(args.sample_configs.sample_k) + '_'
    return extra_path if len(extra_path) else None


def get_transform(args: Union[Namespace, ConfigDict]):
    # I-MLE training does not require transform, instead the masks are given by upstream + I-MLE
    if args.imle_configs is not None:
        return None
    # normal training
    elif args.sample_configs.sample_policy is None:
        return None
    # train with sampling on the fly
    elif args.sample_configs.sample_policy in ['greedy_neighbors', 'topk', 'graph_topk']:
        return policy2transform(args.sample_configs.sample_policy,
                                args.sample_configs.sample_k,
                                args.sample_configs.ensemble,
                                args.sample_configs.subgraph2node_aggr)
    else:
        raise NotImplementedError


def get_pretransform(args: Union[Namespace, ConfigDict], extra_pretransforms: Optional[List] = None):
    pretransform = [GraphToUndirected(), AugmentwithNNodes(), GraphAddRemainSelfLoop()]
    if extra_pretransforms is not None:
        pretransform = pretransform + extra_pretransforms

    if args.sample_configs.sample_policy in ['khop']:
        pretransform.append(policy2transform(args.sample_configs.sample_policy, args.sample_configs.sample_k, 1))

    if hasattr(args.imle_configs, 'emb_spd') and args.imle_configs.emb_spd:
        pretransform.append(AugmentWithShortedPathDistance(MAX_NUM_NODE_DICT[args.dataset.lower()]))

    if hasattr(args.imle_configs, 'emb_ppr') and args.imle_configs.emb_ppr:
        pretransform.append(AugmentWithPPR(MAX_NUM_NODE_DICT[args.dataset.lower()]))

    pretransform = sorted(pretransform, key=lambda p: PRETRANSFORM_PRIORITY[type(p)], reverse=True)
    return Compose(pretransform)


def get_data(args: Union[Namespace, ConfigDict], *_args) -> Tuple[List[AttributedDataLoader],
                                                                  List[AttributedDataLoader],
                                                                  List[AttributedDataLoader]]:
    """
    Distributor function

    :param args:
    :return:
    """
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

    if 'ogbg' in args.dataset.lower():
        train_set, val_set, test_set, mean, std = get_ogbg_data(args)
    elif 'ogbn' in args.dataset.lower():
        train_set, val_set, test_set, mean, std = get_ogbn_data(args)
    elif args.dataset.lower() == 'zinc':
        train_set, val_set, test_set, mean, std = get_zinc(args)
    elif args.dataset.lower() in ['cora', 'pubmed']:
        train_set, val_set, test_set, mean, std = get_planetoid(args)
    elif args.dataset.lower() == 'qm9':
        train_set, val_set, test_set, mean, std = get_qm9(args)
    else:
        raise ValueError

    if isinstance(train_set, list):
        train_loaders = [AttributedDataLoader(
            loader=DataLoader(t,
                              batch_size=args.batch_size,
                              shuffle=not args.debug,
                              num_workers=16),
            mean=mean,
            std=std) for t in train_set]
    elif isinstance(train_set, DATASET):
        train_loaders = [AttributedDataLoader(
            loader=DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=not args.debug,
                              num_workers=16),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    if isinstance(val_set, list):
        val_loaders = [AttributedDataLoader(
            loader=DataLoader(t,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=16),
            mean=mean,
            std=std) for t in val_set]
    elif isinstance(val_set, DATASET):
        val_loaders = [AttributedDataLoader(
            loader=DataLoader(val_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=16),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    if isinstance(test_set, list):
        test_loaders = [AttributedDataLoader(
            loader=DataLoader(t,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=16),
            mean=mean,
            std=std) for t in test_set]
    elif isinstance(test_set, DATASET):
        test_loaders = [AttributedDataLoader(
            loader=DataLoader(test_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=16),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    return train_loaders, val_loaders, test_loaders


def get_ogbg_data(args: Union[Namespace, ConfigDict]):
    pre_transform = get_pretransform(args)
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
    split_idx = dataset.get_idx_split()

    train_idx = split_idx["train"] if not args.debug else split_idx["train"][:16]
    val_idx = split_idx["valid"] if not args.debug else split_idx["valid"][:16]
    test_idx = split_idx["test"] if not args.debug else split_idx["test"][:16]

    train_set = dataset[train_idx]
    val_set = dataset[val_idx]
    test_set = dataset[test_idx]

    return train_set, val_set, test_set, None, None,


def get_ogbn_data(args: Union[Namespace, ConfigDict]):
    pre_transform = get_pretransform(args, extra_pretransforms=[GraphAddRemainSelfLoop()])
    transform = get_transform(args)

    # if there are specific pretransforms, create individual folders for the dataset
    datapath = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    train_set = MyPygNodePropPredDataset(name=args.dataset,
                                         root=datapath,
                                         training_split='train',
                                         khop=args.khop,
                                         transform=transform,
                                         pre_transform=pre_transform)

    val_set = MyPygNodePropPredDataset(name=args.dataset,
                                       root=datapath,
                                       training_split='valid',
                                       khop=args.khop,
                                       transform=transform,
                                       pre_transform=pre_transform)

    test_set = MyPygNodePropPredDataset(name=args.dataset,
                                        root=datapath,
                                        training_split='test',
                                        khop=args.khop,
                                        transform=transform,
                                        pre_transform=pre_transform)

    return train_set, val_set, test_set, None, None,


def get_zinc(args: Union[Namespace, ConfigDict]):
    pre_transform = get_pretransform(args, extra_pretransforms=[
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

    return train_set, val_set, test_set, None, None


def get_planetoid(args: Union[Namespace, ConfigDict]):
    pre_transform = get_pretransform(args, extra_pretransforms=[GraphAddRemainSelfLoop()])
    transform = get_transform(args)

    # if there are specific pretransforms, create individual folders for the dataset
    datapath = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    train = PlanetoidKhopInductive(root=datapath,
                                   name=NAME_DICT[args.dataset.lower()],
                                   khop=args.khop,
                                   training_split='train',
                                   split='public',
                                   transform=transform,
                                   pre_transform=pre_transform)
    val = PlanetoidKhopInductive(root=datapath,
                                 name=NAME_DICT[args.dataset.lower()],
                                 khop=args.khop,
                                 training_split='val',
                                 split='public',
                                 transform=transform,
                                 pre_transform=pre_transform)
    test = PlanetoidKhopInductive(root=datapath,
                                  name=NAME_DICT[args.dataset.lower()],
                                  khop=args.khop,
                                  training_split='test',
                                  split='public',
                                  transform=transform,
                                  pre_transform=pre_transform)

    if args.debug:
        train = train[:16]
        val = val[:16]
        test = test[:16]

    return train, val, test, None, None


def get_qm9(args: Union[Namespace, ConfigDict]):
    pre_transform = get_pretransform(args)
    transform = get_transform(args)

    datapath = os.path.join(args.data_path, 'QM9')
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    train_set = MyQM9(root=datapath,
                      split='train',
                      transform=transform,
                      pre_transform=pre_transform)

    val_set = MyQM9(root=datapath,
                    split='val',
                    transform=transform,
                    pre_transform=pre_transform)

    test_set = MyQM9(root=datapath,
                     split='test',
                     transform=transform,
                     pre_transform=pre_transform)

    train_set.data.y = train_set.data.y[:, 0:12]
    val_set.data.y = val_set.data.y[:, 0:12]
    test_set.data.y = test_set.data.y[:, 0:12]

    # https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch/blob/master/data_loader/data_generator.py#L31
    train_mean = train_set.data.y.mean(dim=0, keepdim=True)
    train_std = train_set.data.y.std(dim=0, keepdim=True)

    train_set.data.y = (train_set.data.y - train_mean) / train_std
    val_set.data.y = (val_set.data.y - train_mean) / train_std
    test_set.data.y = (test_set.data.y - train_mean) / train_std

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    return train_set, val_set, test_set, train_mean, train_std
