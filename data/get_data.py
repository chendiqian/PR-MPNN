import os
from argparse import Namespace
from functools import partial
from typing import Tuple, Union, List, Optional

from ml_collections import ConfigDict
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import DataLoader as PTDataLoader
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import Compose, AddRandomWalkPE, AddLaplacianEigenvectorPE

from .tree_dataset import MyTreeDataset
from .const import DATASET_FEATURE_STAT_DICT, MAX_NUM_NODE_DICT
from .data_preprocess import (GraphExpandDim,
                              GraphToUndirected, GraphCoalesce,
                              GraphCanonicalYClass,
                              AugmentwithNNodes,
                              GraphAttrToOneHot,
                              GraphAddRemainSelfLoop,
                              AugmentWithShortedPathDistance,
                              AugmentWithPPR,
                              AugmentWithDirectedGlobalRewiredGraphs,
                              AugmentWithUndirectedGlobalRewiredGraphs,
                              AugmentWithExtraUndirectedGlobalRewiredGraphs,
                              AugmentWithHybridRewiredGraphs,
                              AugmentWithSpatialInfo,
                              AugmentWithPlotCoordinates,
                              my_collate_fn, collate_fn_with_origin_list)
from .data_utils import AttributedDataLoader

NUM_WORKERS = 0

DATASET = (PygGraphPropPredDataset, ZINC, MyTreeDataset)

# sort keys, some pre_transform should be executed first
PRETRANSFORM_PRIORITY = {
    GraphExpandDim: 0,  # low
    GraphCanonicalYClass: 0,
    GraphAddRemainSelfLoop: 100,  # highest
    GraphToUndirected: 99,  # high
    GraphCoalesce: 99,
    AugmentwithNNodes: 0,  # low
    GraphAttrToOneHot: 0,  # low
    AugmentWithShortedPathDistance: 98,
    AugmentWithPPR: 98,
    AddRandomWalkPE: 98,
    AddLaplacianEigenvectorPE: 98,
    AugmentWithSpatialInfo: 98,
    AugmentWithPlotCoordinates: 98,
}


def get_additional_path(args: Union[Namespace, ConfigDict]):
    extra_path = ''
    if hasattr(args.imle_configs, 'emb_spd') and args.imle_configs.emb_spd:
        extra_path += 'SPDaug_'
    if hasattr(args.imle_configs, 'emb_ppr') and args.imle_configs.emb_ppr:
        extra_path += 'PPRaug_'
    if hasattr(args.imle_configs, 'rwse') or hasattr(args, 'rwse'):
        extra_path += 'rwse_'
    if hasattr(args.imle_configs, 'lap') or hasattr(args, 'lap'):
        extra_path += 'lap_'
    if hasattr(args.imle_configs, 'attenbias'):
        extra_path += 'attenbias_'
    return extra_path if len(extra_path) else None


def get_transform(args: Union[Namespace, ConfigDict]):
    # I-MLE training does not require transform, instead the masks are given by upstream + I-MLE
    if args.imle_configs is not None:
        return None
    # normal training
    if args.sample_configs.sample_policy is None:
        return None
    elif args.sample_configs.sample_policy == 'global_topk_directed':
        transform = AugmentWithDirectedGlobalRewiredGraphs(args.sample_configs.sample_k,
                                                           args.sample_configs.include_original_graph,
                                                           args.sample_configs.ensemble)
    elif args.sample_configs.sample_policy == 'global_topk_undirected':
        transform = AugmentWithUndirectedGlobalRewiredGraphs(args.sample_configs.sample_k,
                                                             args.sample_configs.include_original_graph,
                                                             args.sample_configs.ensemble)
    elif args.sample_configs.sample_policy == 'global_topk_semi':
        transform = AugmentWithExtraUndirectedGlobalRewiredGraphs(args.sample_configs.sample_k,
                                                                  args.sample_configs.include_original_graph,
                                                                  args.sample_configs.ensemble)
    elif args.sample_configs.sample_policy == 'global_topk_hybrid':
        transform = AugmentWithHybridRewiredGraphs(args.sample_configs.sample_k,
                                                   args.sample_configs.include_original_graph,
                                                   args.sample_configs.ensemble)
    else:
        raise ValueError
    return transform


def get_pretransform(args: Union[Namespace, ConfigDict], extra_pretransforms: Optional[List] = None):
    pretransform = [AugmentwithNNodes(), AugmentWithPlotCoordinates()]
    if extra_pretransforms is not None:
        pretransform = pretransform + extra_pretransforms

    if hasattr(args.imle_configs, 'emb_spd') and args.imle_configs.emb_spd:
        pretransform.append(AugmentWithShortedPathDistance(MAX_NUM_NODE_DICT[args.dataset.lower()]))

    if hasattr(args.imle_configs, 'emb_ppr') and args.imle_configs.emb_ppr:
        pretransform.append(AugmentWithPPR(MAX_NUM_NODE_DICT[args.dataset.lower()]))

    if hasattr(args.imle_configs, 'rwse'):
        pretransform.append(AddRandomWalkPE(args.imle_configs.rwse.kernel, 'pestat_RWSE'))
    elif hasattr(args, 'rwse'):
        pretransform.append(AddRandomWalkPE(args.rwse.kernel, 'pestat_RWSE'))

    if hasattr(args.imle_configs, 'lap'):
        pretransform.append(AddLaplacianEigenvectorPE(args.imle_configs.lap.max_freqs, 'EigVecs', is_undirected=True))
    elif hasattr(args, 'lap'):
        pretransform.append(AddLaplacianEigenvectorPE(args.lap.max_freqs, 'EigVecs', is_undirected=True))

    if hasattr(args.imle_configs, 'attenbias'):
        pretransform.append(AugmentWithSpatialInfo(args.imle_configs.attenbias.num_spatial_types,
                                                   args.imle_configs.attenbias.num_in_degrees,
                                                   args.imle_configs.attenbias.num_out_degrees))

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
    elif args.dataset.lower().startswith('tree'):
        train_set, val_set, test_set, mean, std = get_treedataset(args)
    else:
        raise ValueError

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
                             collate_fn=my_collate_fn)
    else:
        # PyG removes the collate function passed in
        dataloader = partial(PyGDataLoader,
                             batch_size=args.batch_size,
                             shuffle=not args.debug,
                             num_workers=NUM_WORKERS)

    if isinstance(train_set, list):
        train_loaders = [AttributedDataLoader(
            loader=dataloader(t),
            mean=mean,
            std=std) for t in train_set]
    elif isinstance(train_set, DATASET):
        train_loaders = [AttributedDataLoader(
            loader=dataloader(train_set),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    if isinstance(val_set, list):
        val_loaders = [AttributedDataLoader(
            loader=dataloader(t),
            mean=mean,
            std=std) for t in val_set]
    elif isinstance(val_set, DATASET):
        val_loaders = [AttributedDataLoader(
            loader=dataloader(val_set),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    if isinstance(test_set, list):
        test_loaders = [AttributedDataLoader(
            loader=dataloader(t),
            mean=mean,
            std=std) for t in test_set]
    elif isinstance(test_set, DATASET):
        test_loaders = [AttributedDataLoader(
            loader=dataloader(test_set),
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

    return train_set, val_set, test_set, None, None


def get_treedataset(args: Union[Namespace, ConfigDict]):
    pre_transform = get_pretransform(args, extra_pretransforms=[GraphCoalesce(), GraphCanonicalYClass()])
    transform = get_transform(args)

    depth = int(args.dataset.lower().split('_')[1])
    assert 2 <= depth <= 8
    data_path = os.path.join(args.data_path, args.dataset)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = MyTreeDataset(data_path, True, 11, depth, transform=transform, pre_transform=pre_transform)
    val_set = MyTreeDataset(data_path, False, 11, depth, transform=transform, pre_transform=pre_transform)
    test_set = val_set

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    return train_set, val_set, test_set, None, None
