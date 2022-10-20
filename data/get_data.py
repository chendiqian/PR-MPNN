import os
from typing import Tuple, Union, List
from argparse import Namespace
from ml_collections import ConfigDict

from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset

from data.data_utils import AttributedDataLoader
from data.data_preprocess import GraphToUndirected, GraphCoalesce, AugmentwithNNodes, policy2transform

DATASET = (PygGraphPropPredDataset,)


def get_transform(args: Union[Namespace, ConfigDict]):
    # I-MLE training does not require transform, instead the masks are given by upstream + I-MLE
    if args.imle_configs is not None:
        return None
    # normal training
    elif args.sample_configs.sample_policy is None:
        return None
    # train with sampling on the fly
    elif args.sample_configs.sample_policy in ['greedy_neighbors']:
        return policy2transform(args.sample_configs.sample_policy,
                                args.sample_configs.sample_k)


def get_pretransform(args: Union[Namespace, ConfigDict]):
    pretransform = [GraphToUndirected(), GraphCoalesce(), AugmentwithNNodes()]
    extra_path = None
    if args.sample_configs.sample_policy in ['khop']:
        pretransform.append(policy2transform(args.sample_configs.sample_policy, args.sample_configs.sample_k))
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

    if 'ogb' in args.dataset.lower():
        train_set, val_set, test_set, mean, std = get_ogb_data(args)
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


def get_ogb_data(args: Union[Namespace, ConfigDict]):
    pre_transform, extra_path = get_pretransform(args)
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
