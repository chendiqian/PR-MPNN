import torch.nn as nn

from data.const import DATASET_FEATURE_STAT_DICT, TYPE_ENCODER
from models.downstream import GNN_Duo
from models.hybrid_model import HybridModel
from models.my_encoder import FeatureEncoder, BondEncoder, QM9FeatureEncoder
from models.rewirer import GraphRewirer
from models.upstream import EdgeSelector
from simple.simple_scheme import EdgeSIMPLEBatched


def get_encoder(args, hidden):
    type_encoder = TYPE_ENCODER[args.dataset.lower()]
    lap = args.lap if hasattr(args, 'lap') else None
    rwse = args.rwse if hasattr(args, 'rwse') else None

    if 'qm9' in args.dataset.lower():
        encoder = QM9FeatureEncoder(DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                                    hidden,
                                    type_encoder,
                                    lap,
                                    rwse)
    else:
        encoder = FeatureEncoder(
            dim_in=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
            hidden=hidden,
            type_encoder=type_encoder,
            lap_encoder=lap,
            rw_encoder=rwse)

    if args.dataset.lower() in ['zinc', 'alchemy', 'edge_wt_region_boundary', 'qm9',
                                'ppgnqm9', 'exp', 'cexp', 'sym_skipcircles', 'ptc_mr', 'mutag', 'csl',
                                'imdb-m', 'imdb-b']:
        edge_encoder = nn.Sequential(
            nn.Linear(DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'], hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden))
    elif args.dataset.lower().startswith('hetero') or \
            args.dataset.lower().startswith('tree') or \
            (args.dataset.lower().startswith('sym') and args.dataset.lower != 'sym_skipcircles') or \
            args.dataset.lower().startswith('leafcolor') or \
            args.dataset.lower() == 'proteins' or \
            args.dataset.lower() == 'nci1' or \
            args.dataset.lower() == 'nci109':
        edge_encoder = None
    elif args.dataset.lower().startswith('peptides') or \
            args.dataset.lower().startswith('ogbg'):
        edge_encoder = BondEncoder(hidden)
    else:
        raise NotImplementedError("we need torch.nn.Embedding, to be implemented")

    return encoder, edge_encoder


def get_model(args, *_args):
    # downstream model
    encoder, edge_encoder = get_encoder(args, args.hid_size)
    model = GNN_Duo(encoder,
                    edge_encoder,
                    args.model.lower(),
                    include_org=args.sample_configs.include_original_graph,
                    num_candidates=2 if args.sample_configs.separate and
                                        args.sample_configs.sample_k > 0 and
                                        args.sample_configs.sample_k2 > 0 else 1,
                    num_layers=args.num_convlayers,
                    hidden=args.hid_size,
                    num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                    use_bn=args.bn,
                    dropout=args.dropout,
                    residual=args.residual,
                    mlp_layers_intragraph=args.mlp_layers_intragraph,
                    mlp_layers_intergraph=args.mlp_layers_intergraph,
                    graph_pooling=args.graph_pooling,
                    inter_graph_pooling=args.inter_graph_pooling)

    # upstream model
    # not shared with the downstream
    encoder, edge_encoder = get_encoder(args, args.imle_configs.emb_hid_size)
    emb_model = EdgeSelector(encoder,
                             edge_encoder,
                             in_dim=args.imle_configs.emb_hid_size,
                             hid_size=args.imle_configs.emb_hid_size,
                             gnn_layer=args.imle_configs.gnn_layer,
                             mlp_layer=args.imle_configs.mlp_layer,
                             use_deletion_head=True,
                             directed_sampling=args.sample_configs.directed,
                             dropout=args.dropout,
                             ensemble=args.sample_configs.ensemble,
                             use_bn=args.imle_configs.batchnorm)

    # rewiring process
    simple_sampler = EdgeSIMPLEBatched()

    # we recommend using `SimplifiedGraphRewirer` for its efficiency
    # if there is customized required, e.g., need to un-collate the rewired batch
    # into graphs for plots, you need to use GraphRewirer
    rewiring = GraphRewirer(args.sample_configs.sample_k,
                            args.sample_configs.sample_k2,
                            args.imle_configs.num_train_ensemble,
                            args.imle_configs.num_val_ensemble,
                            simple_sampler,
                            args.sample_configs.separate,
                            args.sample_configs.directed,
                            args.imle_configs.auxloss
                            if hasattr(args.imle_configs, 'auxloss') else None)

    hybrid_model = HybridModel(emb_model, model, rewiring)
    return hybrid_model
