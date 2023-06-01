import os
from argparse import Namespace
from functools import partial
from typing import Tuple, Union, List, Optional

from ml_collections import ConfigDict
from networkx import kamada_kawai_layout
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import DataLoader as PTDataLoader
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import Compose, AddRandomWalkPE, AddLaplacianEigenvectorPE

from .const import DATASET_FEATURE_STAT_DICT, MAX_NUM_NODE_DICT
from .data_preprocess import (GraphExpandDim,
                              GraphToUndirected, GraphCoalesce,
                              GraphCanonicalYClass,
                              AugmentwithNumbers,
                              GraphAttrToOneHot,
                              GraphAddRemainSelfLoop, GraphAddSkipConnection,
                              GraphRedirect,
                              AugmentWithShortedPathDistance,
                              AugmentWithEdgeCandidate,
                              AugmentWithPPR,
                              AugmentWithRandomRewiredGraphs,
                              AugmentWithPlotCoordinates,
                              make_collate4random_baseline, collate_fn_with_origin_list)
from .data_utils import AttributedDataLoader, circular_tree_layout
from .heterophilic import HeterophilicDataset
from .peptides_func import PeptidesFunctionalDataset
from .peptides_struct import PeptidesStructuralDataset
from .tree_dataset import MyTreeDataset, MyLeafColorDataset
from .tudataset import MyTUDataset
from .voc_superpixels import VOCSuperpixels

NUM_WORKERS = 0

DATASET = (PygGraphPropPredDataset,
           ZINC,
           MyTreeDataset,
           MyLeafColorDataset,
           MyTUDataset,
           PeptidesStructuralDataset,
           PeptidesFunctionalDataset,
           VOCSuperpixels,
           HeterophilicDataset)

# sort keys, some pre_transform should be executed first
PRETRANSFORM_PRIORITY = {
    GraphExpandDim: 0,  # low
    GraphCanonicalYClass: 0,
    GraphAddRemainSelfLoop: 100,  # highest
    GraphAddSkipConnection: 100,
    GraphRedirect: 100,
    GraphToUndirected: 99,  # high
    GraphCoalesce: 99,
    AugmentwithNumbers: 0,  # low
    GraphAttrToOneHot: 0,  # low
    AugmentWithShortedPathDistance: 98,
    AugmentWithEdgeCandidate: 98,
    AugmentWithPPR: 98,
    AddRandomWalkPE: 98,
    AddLaplacianEigenvectorPE: 98,
    AugmentWithPlotCoordinates: 98,
}


def get_additional_path(args: Union[Namespace, ConfigDict]):
    extra_path = ''
    if args.sample_configs.sample_policy is not None and 'edge_candid' in args.sample_configs.sample_policy:
        heu = args.sample_configs.heuristic if hasattr(args.sample_configs, 'heuristic') else 'longest_path'
        directed = args.sample_configs.directed if hasattr(args.sample_configs, 'directed') else False
        extra_path += f'EdgeCandidates_{heu}_{"dir" if directed else "undir"}_{args.sample_configs.candid_pool}_'
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
    elif args.sample_configs.sample_policy == 'add_del':
        raise NotImplementedError("need to implement directedness")
        # transform = AugmentWithRandomRewiredGraphs(sample_k_add=args.sample_configs.sample_k,
        #                                            sample_k_del=args.sample_configs.sample_k2,
        #                                            include_original_graph=args.sample_configs.include_original_graph,
        #                                            in_place=args.sample_configs.in_place,
        #                                            ensemble=args.sample_configs.ensemble)
    else:
        raise ValueError


def get_pretransform(args: Union[Namespace, ConfigDict], extra_pretransforms: Optional[List] = None):
    pretransform = [AugmentwithNumbers()]
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

    # add edge candidates or bidirectional
    if args.sample_configs.sample_policy is not None and 'edge_candid' in args.sample_configs.sample_policy:
        heu = args.sample_configs.heuristic if hasattr(args.sample_configs, 'heuristic') else 'longest_path'
        directed = args.sample_configs.directed if hasattr(args.sample_configs, 'directed') else False
        pretransform.append(AugmentWithEdgeCandidate(heu, args.sample_configs.candid_pool, directed))

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

    task = 'graph'
    if 'ogbg' in args.dataset.lower():
        train_set, val_set, test_set, mean, std = get_ogbg_data(args)
    elif args.dataset.lower() == 'zinc':
        train_set, val_set, test_set, mean, std = get_zinc(args)
    elif args.dataset.lower() == 'alchemy':
        train_set, val_set, test_set, mean, std = get_alchemy(args)
    elif args.dataset.lower().startswith('tree'):
        train_set, val_set, test_set, mean, std = get_treedataset(args)
    elif args.dataset.lower().startswith('leafcolor'):
        train_set, val_set, test_set, mean, std = get_leafcolordataset(args)
    elif args.dataset.lower().startswith('peptides-struct'):
        train_set, val_set, test_set, mean, std = get_peptides(args, set='struct')
    elif args.dataset.lower() == 'edge_wt_region_boundary':
        train_set, val_set, test_set, mean, std = get_vocsuperpixel(args)
        task = 'node'
    elif args.dataset.lower().startswith('hetero'):
        train_set, val_set, test_set, mean, std = get_heterophily(args)
        task = 'node'
    elif args.dataset.lower().startswith('peptides-func'):
        train_set, val_set, test_set, mean, std = get_peptides(args, set='func')

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
                             collate_fn=make_collate4random_baseline(
                                 include_org=args.sample_configs.include_original_graph
                             ))
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
            std=std,
            task=task) for t in train_set]
    elif isinstance(train_set, DATASET):
        train_loaders = [AttributedDataLoader(
            loader=dataloader(train_set),
            mean=mean,
            std=std,
            task=task)]
    else:
        raise TypeError

    if isinstance(val_set, list):
        val_loaders = [AttributedDataLoader(
            loader=dataloader(t),
            mean=mean,
            std=std,
            task=task) for t in val_set]
    elif isinstance(val_set, DATASET):
        val_loaders = [AttributedDataLoader(
            loader=dataloader(val_set),
            mean=mean,
            std=std,
            task=task)]
    else:
        raise TypeError

    if isinstance(test_set, list):
        test_loaders = [AttributedDataLoader(
            loader=dataloader(t),
            mean=mean,
            std=std,
            task=task) for t in test_set]
    elif isinstance(test_set, DATASET):
        test_loaders = [AttributedDataLoader(
            loader=dataloader(test_set),
            mean=mean,
            std=std,
            task=task)]
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

    return train_set, val_set, test_set, None, None

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
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    return train_set, val_set, test_set, None, None

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

    return train_set, val_set, test_set, mean, std


def get_treedataset(args: Union[Namespace, ConfigDict]):
    depth = int(args.dataset.lower().split('_')[1])
    assert 2 <= depth <= 8

    pre_transform = get_pretransform(args, extra_pretransforms=[GraphCoalesce(), GraphCanonicalYClass(), AugmentWithPlotCoordinates(layout=circular_tree_layout)])
    # pre_transform = get_pretransform(args, extra_pretransforms=[GraphCoalesce(), GraphCanonicalYClass(), GraphRedirect(depth)])
    transform = get_transform(args)

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

    return train_set, val_set, test_set, None, None


def get_vocsuperpixel(args):
    datapath = os.path.join(args.data_path, 'VOCSuperpixels')
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)
    pre_transform = get_pretransform(args, extra_pretransforms=None)
    transform = get_transform(args)

    splits = [VOCSuperpixels(root=datapath,
                             name=args.dataset.lower(),
                             split=sp,
                             transform=transform,
                             pre_transform=pre_transform) for sp in ['train', 'val', 'test']]

    train_set, val_set, test_set = splits

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    return train_set, val_set, test_set, None, None

def get_heterophily(args):
    dataset_name = args.dataset.lower().split('_')[1]
    datapath = os.path.join(args.data_path, 'hetero_' + dataset_name)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    pre_transforms = get_pretransform(args, extra_pretransforms=[GraphToUndirected()])
    transform = get_transform(args)

    splits = [[HeterophilicDataset(root=datapath,
                                   name=dataset_name,
                                   split=split,
                                   fold=fold,
                                   transform=transform,
                                   pre_transform=pre_transforms) for fold in range(10)] for split in ['train', 'val', 'test']]

    train_set, val_set, test_set = splits

    if args.debug:
        train_set = train_set[0]
        val_set = val_set[0]
        test_set = test_set[0]

    return train_set, val_set, test_set, None, None
