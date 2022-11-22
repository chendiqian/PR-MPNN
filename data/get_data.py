import os
from typing import Tuple, Union, List
from argparse import Namespace
from ml_collections import ConfigDict

from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, TUDataset
from ogb.graphproppred import PygGraphPropPredDataset

from .data_utils import AttributedDataLoader
from .data_preprocess import (GraphExpandDim,
                              GraphToUndirected,
                              AugmentwithNNodes,
                              policy2transform,
                              GraphAttrToOneHot,
                              GraphAddRemainSelfLoop)
from .const import DATASET_FEATURE_STAT_DICT
from .custom_dataset import PlanetoidKhopInductive

DATASET = (PygGraphPropPredDataset, ZINC, TUDataset, PlanetoidKhopInductive)

NAME_DICT = {'zinc_full': "ZINC_full",
             'cora': 'Cora',
             'pubmed': 'PubMed'}


def get_transform(args: Union[Namespace, ConfigDict]):
    # I-MLE training does not require transform, instead the masks are given by upstream + I-MLE
    if args.imle_configs is not None:
        return None
    # normal training
    elif args.sample_configs.sample_policy is None:
        return None
    # train with sampling on the fly
    elif args.sample_configs.sample_policy in ['greedy_neighbors', 'topk']:
        return policy2transform(args.sample_configs.sample_policy,
                                args.sample_configs.sample_k,
                                args.sample_configs.ensemble)
    else:
        raise NotImplementedError


def get_pretransform(args: Union[Namespace, ConfigDict], extra_pretransforms: List, extra_post: bool = True):
    pretransform = [GraphToUndirected(), AugmentwithNNodes()]
    if extra_post:
        pretransform = pretransform + extra_pretransforms
    else:
        pretransform = extra_pretransforms + pretransform
    extra_path = None
    if args.sample_configs.sample_policy in ['khop']:
        pretransform.append(policy2transform(args.sample_configs.sample_policy, args.sample_configs.sample_k, 1))
        extra_path = args.sample_configs.sample_policy + '_' + str(args.sample_configs.sample_k)
    return Compose(pretransform), extra_path


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
    elif args.dataset.lower() == 'zinc':
        train_set, val_set, test_set, mean, std = get_zinc(args)
    elif args.dataset.lower() == 'zinc_full':
        train_set, val_set, test_set, mean, std = get_TUdata(args)
    elif args.dataset.lower() in ['cora', 'pubmed']:
        train_set, val_set, test_set, mean, std = get_planetoid(args)
    else:
        raise ValueError

    if isinstance(train_set, list):
        train_loaders = [AttributedDataLoader(
            loader=DataLoader(t,
                              batch_size=args.batch_size,
                              shuffle=not args.debug, ),
            mean=mean,
            std=std) for t in train_set]
    elif isinstance(train_set, DATASET):
        train_loaders = [AttributedDataLoader(
            loader=DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=not args.debug, ),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    if isinstance(val_set, list):
        val_loaders = [AttributedDataLoader(
            loader=DataLoader(t,
                              batch_size=args.batch_size,
                              shuffle=False, ),
            mean=mean,
            std=std) for t in val_set]
    elif isinstance(val_set, DATASET):
        val_loaders = [AttributedDataLoader(
            loader=DataLoader(val_set,
                              batch_size=args.batch_size,
                              shuffle=False, ),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    if isinstance(test_set, list):
        test_loaders = [AttributedDataLoader(
            loader=DataLoader(t,
                              batch_size=args.batch_size,
                              shuffle=False, ),
            mean=mean,
            std=std) for t in test_set]
    elif isinstance(test_set, DATASET):
        test_loaders = [AttributedDataLoader(
            loader=DataLoader(test_set,
                              batch_size=args.batch_size,
                              shuffle=False, ),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    return train_loaders, val_loaders, test_loaders


def get_ogbg_data(args: Union[Namespace, ConfigDict]):
    pre_transform, extra_path = get_pretransform(args, extra_pretransforms=[])
    transform = get_transform(args)

    # if there are specific pretransforms, create individual folders for the dataset
    datapath = args.data_path
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


def get_zinc(args: Union[Namespace, ConfigDict]):
    pre_transform, extra_path = get_pretransform(args, extra_pretransforms=[
        GraphExpandDim(),
        GraphAttrToOneHot(DATASET_FEATURE_STAT_DICT['zinc']['node'],
                          DATASET_FEATURE_STAT_DICT['zinc']['edge'])])
    transform = get_transform(args)

    data_path = os.path.join(args.data_path, 'ZINC')
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


def get_TUdata(args: Union[Namespace, ConfigDict]):
    """
    :param args
    :return:
    """
    infile = open(f"./datasets/indices/{args.dataset.lower()}/test.index.txt", "r")
    line = next(iter(infile))
    test_indices = line.split(",")
    if args.debug:
        test_indices = test_indices[:16]
    test_indices = [int(i) for i in test_indices]

    infile = open(f"./datasets/indices/{args.dataset.lower()}/val.index.txt", "r")
    line = next(iter(infile))
    val_indices = line.split(",")
    if args.debug:
        val_indices = val_indices[:16]
    val_indices = [int(i) for i in val_indices]

    infile = open(f"./datasets/indices/{args.dataset.lower()}/train.index.txt", "r")
    line = next(iter(infile))
    train_indices = line.split(",")
    if args.debug:
        train_indices = train_indices[:16]
    train_indices = [int(i) for i in train_indices]

    pre_transform, extra_path = get_pretransform(args, extra_pretransforms=[GraphExpandDim()])
    transform = get_transform(args)

    data_path = args.data_path
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    dataset = TUDataset(data_path,
                        name=NAME_DICT[args.dataset.lower()],
                        transform=transform,
                        pre_transform=pre_transform)

    train_set = dataset[:220011][train_indices]
    val_set = dataset[225011:][val_indices]
    test_set = dataset[220011:225011][test_indices]

    return train_set, val_set, test_set, None, None


def get_planetoid(args: Union[Namespace, ConfigDict]):
    pre_transform, extra_path = get_pretransform(args, extra_pretransforms=[GraphAddRemainSelfLoop()], extra_post=False)
    transform = get_transform(args)

    # if there are specific pretransforms, create individual folders for the dataset
    datapath = args.data_path
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
