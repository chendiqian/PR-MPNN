import os
from argparse import Namespace
from typing import Tuple, Union, List, Optional

from ml_collections import ConfigDict
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from .const import DATASET_FEATURE_STAT_DICT, MAX_NUM_NODE_DICT
from .data_preprocess import (GraphExpandDim,
                              GraphToUndirected,
                              AugmentwithNNodes,
                              GraphAttrToOneHot,
                              GraphAddRemainSelfLoop,
                              AugmentWithShortedPathDistance,
                              AugmentWithPPR,
                              AugmentWithRandomWalkProbs, AugmentWithLaplace,
                              AugmentWithPerNodeRewiredGraphs,
                              AugmentWithGlobalRewiredGraphs)
from .data_utils import AttributedDataLoader
NUM_WORKERS = 0

DATASET = (PygGraphPropPredDataset, ZINC)

NAME_DICT = {'zinc_full': "ZINC_full",}

# sort keys, some pre_transform should be executed first
PRETRANSFORM_PRIORITY = {
    GraphExpandDim: 0,  # low
    GraphAddRemainSelfLoop: 100,  # highest
    GraphToUndirected: 99,  # high
    AugmentwithNNodes: 0,  # low
    GraphAttrToOneHot: 0,  # low
    AugmentWithShortedPathDistance: 98,
    AugmentWithPPR: 98,
    AugmentWithRandomWalkProbs: 98,
    AugmentWithLaplace: 98,
}


def get_additional_path(args: Union[Namespace, ConfigDict]):
    extra_path = ''
    if hasattr(args.imle_configs, 'emb_spd') and args.imle_configs.emb_spd:
        extra_path += 'SPDaug_'
    if hasattr(args.imle_configs, 'emb_ppr') and args.imle_configs.emb_ppr:
        extra_path += 'PPRaug_'
    if hasattr(args.imle_configs, 'rwse'):
        extra_path += 'rwse_'
    if hasattr(args.imle_configs, 'lap'):
        extra_path += 'lap_'
    return extra_path if len(extra_path) else None


def get_transform(args: Union[Namespace, ConfigDict]):
    # I-MLE training does not require transform, instead the masks are given by upstream + I-MLE
    if args.imle_configs is not None:
        return None
    # normal training
    if args.sample_configs.sample_policy is None:
        return None
    elif args.sample_configs.sample_policy == 'graph_topk':
        transform = AugmentWithPerNodeRewiredGraphs(args.sample_configs.sample_k,
                                                    args.sample_configs.include_original_graph,
                                                    args.sample_configs.ensemble)
    elif args.sample_configs.sample_policy == 'global_topk':
        transform = AugmentWithGlobalRewiredGraphs(args.sample_configs.sample_k,
                                                   args.sample_configs.include_original_graph,
                                                   args.sample_configs.ensemble)
    else:
        raise ValueError
    return transform


def get_pretransform(args: Union[Namespace, ConfigDict], extra_pretransforms: Optional[List] = None):
    pretransform = [GraphToUndirected(), AugmentwithNNodes(), GraphAddRemainSelfLoop()]
    if extra_pretransforms is not None:
        pretransform = pretransform + extra_pretransforms

    if hasattr(args.imle_configs, 'emb_spd') and args.imle_configs.emb_spd:
        pretransform.append(AugmentWithShortedPathDistance(MAX_NUM_NODE_DICT[args.dataset.lower()]))

    if hasattr(args.imle_configs, 'emb_ppr') and args.imle_configs.emb_ppr:
        pretransform.append(AugmentWithPPR(MAX_NUM_NODE_DICT[args.dataset.lower()]))

    if hasattr(args.imle_configs, 'rwse'):
        pretransform.append(AugmentWithRandomWalkProbs(eval(args.imle_configs.rwse.kernel)))

    if hasattr(args.imle_configs, 'lap'):
        pretransform.append(AugmentWithLaplace(args.imle_configs.lap.eigen.laplacian_norm,
                                               args.imle_configs.lap.eigen.max_freqs,
                                               args.imle_configs.lap.eigen.eigvec_norm))

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
    elif args.dataset.lower() == 'zinc':
        train_set, val_set, test_set, mean, std = get_zinc(args)
    else:
        raise ValueError

    if isinstance(train_set, list):
        train_loaders = [AttributedDataLoader(
            loader=DataLoader(t,
                              batch_size=args.batch_size,
                              shuffle=not args.debug,
                              num_workers=NUM_WORKERS),
            mean=mean,
            std=std) for t in train_set]
    elif isinstance(train_set, DATASET):
        train_loaders = [AttributedDataLoader(
            loader=DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=not args.debug,
                              num_workers=NUM_WORKERS),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    if isinstance(val_set, list):
        val_loaders = [AttributedDataLoader(
            loader=DataLoader(t,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=NUM_WORKERS),
            mean=mean,
            std=std) for t in val_set]
    elif isinstance(val_set, DATASET):
        val_loaders = [AttributedDataLoader(
            loader=DataLoader(val_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=NUM_WORKERS),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    if isinstance(test_set, list):
        test_loaders = [AttributedDataLoader(
            loader=DataLoader(t,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=NUM_WORKERS),
            mean=mean,
            std=std) for t in test_set]
    elif isinstance(test_set, DATASET):
        test_loaders = [AttributedDataLoader(
            loader=DataLoader(test_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=NUM_WORKERS),
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
