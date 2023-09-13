import os
from argparse import Namespace
from functools import partial
from typing import Union, List, Optional

import torch
from ml_collections import ConfigDict
from torch.utils.data import DataLoader as PTDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import Compose, AddRandomWalkPE, AddLaplacianEigenvectorPE
from torch_geometric.utils import degree as pyg_degree

from data.lrgb import MyLRGBDataset
from data.utils.datatype_utils import AttributedDataLoader
from .data_preprocess import (GraphToUndirected, GraphCoalesce,
                              AugmentwithNumbers,
                              GraphAttrToOneHot,
                              GraphAddRemainSelfLoop,
                              AugmentWithEdgeCandidate,
                              AugmentWithPlotCoordinates,
                              IncTransform,
                              collate_fn_with_origin_list)
from .random_baseline import AugmentWithRandomRewiredGraphs, collate_random_rewired_batch

NUM_WORKERS = 0

# sort keys, some pre_transform should be executed first
PRETRANSFORM_PRIORITY = {
    GraphAddRemainSelfLoop: 100,  # highest
    GraphToUndirected: 99,  # high
    GraphCoalesce: 99,
    AugmentwithNumbers: 0,  # low
    GraphAttrToOneHot: 0,  # low
    AugmentWithEdgeCandidate: 98,
    AddRandomWalkPE: 98,
    AddLaplacianEigenvectorPE: 98,
    AugmentWithPlotCoordinates: 98,
}


def get_additional_path(args: Union[Namespace, ConfigDict]):
    extra_path = ''
    if args.sample_configs.sample_policy == 'edge_candid':
        heu = args.sample_configs.heuristic if hasattr(args.sample_configs,
                                                       'heuristic') else 'longest_path'
        directed = args.sample_configs.directed if hasattr(args.sample_configs,
                                                           'directed') else False
        extra_path += f'EdgeCandidates_{heu}_{"dir" if directed else "undir"}_{args.sample_configs.candid_pool}_'
    if hasattr(args.imle_configs, 'rwse') or hasattr(args, 'rwse'):
        extra_path += 'rwse_'
    if hasattr(args.imle_configs, 'lap') or hasattr(args, 'lap'):
        extra_path += 'lap_'
    return extra_path if len(extra_path) else None


def get_transform(args: Union[Namespace, ConfigDict]):
    # I-MLE training does not require transform, instead the masks are given by upstream + I-MLE
    if args.imle_configs is not None:
        if hasattr(args.imle_configs,
                   'model') and args.imle_configs.model == '2wl_edge_selector':
            return IncTransform()
        else:
            return None
    # normal training
    if args.sample_configs.sample_policy is None:
        return None
    elif args.sample_configs.sample_policy == 'add_del':
        transform = AugmentWithRandomRewiredGraphs(
            sample_k_add=args.sample_configs.sample_k,
            sample_k_del=args.sample_configs.sample_k2,
            include_original_graph=args.sample_configs.include_original_graph,
            in_place=args.sample_configs.in_place,
            ensemble=args.sample_configs.ensemble,
            separate=args.sample_configs.separate,
            directed=args.sample_configs.directed,
            )
        return transform
    else:
        raise ValueError


def get_pretransform(args: Union[Namespace, ConfigDict],
                     extra_pretransforms: Optional[List] = None):
    pretransform = [AugmentwithNumbers()]
    if extra_pretransforms is not None:
        pretransform = pretransform + extra_pretransforms

    if hasattr(args.imle_configs, 'rwse'):
        pretransform.append(AddRandomWalkPE(args.imle_configs.rwse.kernel, 'pestat_RWSE'))
    elif hasattr(args, 'rwse'):
        pretransform.append(AddRandomWalkPE(args.rwse.kernel, 'pestat_RWSE'))

    if hasattr(args.imle_configs, 'lap'):
        pretransform.append(
            AddLaplacianEigenvectorPE(args.imle_configs.lap.max_freqs, 'EigVecs',
                                      is_undirected=True))
    elif hasattr(args, 'lap'):
        pretransform.append(
            AddLaplacianEigenvectorPE(args.lap.max_freqs, 'EigVecs', is_undirected=True))

    # add edge candidates or bidirectional
    if args.sample_configs.sample_policy == 'edge_candid':
        heu = args.sample_configs.heuristic if hasattr(args.sample_configs,
                                                       'heuristic') else 'longest_path'
        directed = args.sample_configs.directed if hasattr(args.sample_configs,
                                                           'directed') else False
        pretransform.append(
            AugmentWithEdgeCandidate(heu, args.sample_configs.candid_pool, directed))

    pretransform = sorted(pretransform, key=lambda p: PRETRANSFORM_PRIORITY[type(p)],
                          reverse=True)
    return Compose(pretransform)


def get_data(args: Union[Namespace, ConfigDict], *_args):
    """
    Distributor function

    :param args:
    :return:
    """
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

    train_set, val_set, test_set, std = get_pcqm(args)
    task = 'edge'

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

    train_loaders = [AttributedDataLoader(
        loader=dataloader(train_set),
        std=std,
        task=task)]

    val_loaders = [AttributedDataLoader(
        loader=dataloader(val_set),
        std=std,
        task=task)]

    test_loaders = [AttributedDataLoader(
        loader=dataloader(test_set),
        std=std,
        task=task)]

    return train_loaders, val_loaders, test_loaders, None



def get_pcqm(args: Union[Namespace, ConfigDict]):
    datapath = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)
    pre_transform = get_pretransform(args, extra_pretransforms=[AugmentwithNumbers()])
    transform = get_transform(args)

    train_set = MyLRGBDataset(name='PCQM-Contact', root=datapath, split='train',
                              transform=transform, pre_transform=pre_transform)
    val_set = MyLRGBDataset(name='PCQM-Contact', root=datapath, split='val',
                            transform=transform, pre_transform=pre_transform)
    test_set = MyLRGBDataset(name='PCQM-Contact', root=datapath, split='test',
                             transform=transform, pre_transform=pre_transform)

    train_set.data.y = train_set.data.y[:, None].float()
    val_set.data.y = val_set.data.y[:, None].float()
    test_set.data.y = test_set.data.y[:, None].float()

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    return train_set, val_set, test_set, None
