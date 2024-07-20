import torch.nn as nn
from data.const import DATASET_FEATURE_STAT_DICT
from models.downstream_models.gnn_duo import GNN_Duo
from models.my_encoder import FeatureEncoder, BondEncoder, QM9FeatureEncoder
from models.upstream_models.edge_candidate_selector import EdgeSelector
from models.downstream_models.qm9_gnn import QM9_Net


def get_encoder(args, for_downstream):
    if args.dataset.lower() in ['zinc', 'alchemy', 'edge_wt_region_boundary', 'qm9', 'exp', 
                                'cexp', 'proteins', 'mutag', 'ptc_mr', 'nci1', 'nci109', 'csl', 'imdb-m', 'imdb-b']:
        type_encoder = 'linear'
    elif args.dataset.lower().startswith('hetero'):
        type_encoder = 'linear'
    elif args.dataset.lower().startswith('sym'):
        type_encoder = 'linear'
    elif args.dataset.lower().startswith('tree'):
        type_encoder = 'bi_embedding'
    elif args.dataset.lower().startswith('leafcolor'):
        type_encoder = 'bi_embedding_cat'
    elif args.dataset.lower().startswith('peptides') or args.dataset.lower().startswith('ogbg'):
        type_encoder = 'atomencoder'
    elif args.dataset.lower() == 'ppgnqm9':
        type_encoder = 'atomencoder'
    else:
        raise ValueError

    # some args vary from downstream and upstream
    if for_downstream:
        hidden = args.hid_size
        lap = args.lap if hasattr(args, 'lap') else None
        rwse = args.rwse if hasattr(args, 'rwse') else None
    else:
        hidden = args.imle_configs.emb_hid_size
        lap = args.imle_configs.lap if hasattr(args.imle_configs, 'lap') else None
        rwse = args.imle_configs.rwse if hasattr(args.imle_configs, 'rwse') else None
    edge_hidden = hidden

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
            nn.Linear(DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'], edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden))
    elif args.dataset.lower().startswith('hetero') or \
            args.dataset.lower().startswith('tree') or \
            (args.dataset.lower().startswith('sym') and args.dataset.lower != 'sym_skipcircles') or \
            args.dataset.lower().startswith('leafcolor') or \
            args.dataset.lower() == 'proteins' or \
            args.dataset.lower() == 'nci1' or  \
            args.dataset.lower() == 'nci109':
        edge_encoder = None
    elif args.dataset.lower().startswith('peptides') or \
            args.dataset.lower().startswith('ogbg'):
        edge_encoder = BondEncoder(edge_hidden)
    else:
        raise NotImplementedError("we need torch.nn.Embedding, to be implemented")

    return encoder, edge_encoder


def get_model(args, device, *_args):
    model, emb_model = None, None
    encoder, edge_encoder = get_encoder(args, for_downstream=True)
    if 'duo' in args.model.lower():
        gnn_type = args.model.lower().split('_')[0]
        if gnn_type.startswith('qm9'):
            assert not (hasattr(args.imle_configs, 'rwse') or hasattr(args, 'rwse')
                        or hasattr(args.imle_configs, 'lap') or hasattr(args, 'lap')), "Need a new node encoder!"

        model = GNN_Duo(encoder,
                        edge_encoder,
                        gnn_type,
                        include_org=args.sample_configs.include_original_graph,
                        num_candidates=2 if args.sample_configs.separate and args.sample_configs.sample_k2 > 0 else 1,
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
    elif args.model in ['qm9_gin', 'qm9_gine']:
        assert not (hasattr(args.imle_configs, 'rwse') or hasattr(args, 'rwse')
                    or hasattr(args.imle_configs, 'lap') or hasattr(args, 'lap')), "Need a new node encoder!"
        model = QM9_Net(
            encoder=encoder,
            gnn_type=args.model.split('_')[-1],
            edge_encoder=edge_encoder,
            num_classes=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['num_class'],
            emb_sizes=args.hid_size,
            num_layers=args.num_convlayers,
            drpt_prob=args.dropout,
            graph_pooling=args.graph_pooling)
    else:
        raise NotImplementedError

    if args.imle_configs is not None and args.imle_configs.model is not None:
        # not shared with the downstream
        encoder, edge_encoder = get_encoder(args, for_downstream=False)
        if args.imle_configs.model == 'edge_selector':
            emb_model = EdgeSelector(encoder,
                                     edge_encoder,
                                     in_dim=args.imle_configs.emb_hid_size,
                                     hid_size=args.imle_configs.emb_hid_size,
                                     gnn_layer=args.imle_configs.gnn_layer,
                                     mlp_layer=args.imle_configs.mlp_layer,
                                     use_deletion_head=True,
                                     directed_sampling=args.sample_configs.directed,
                                     dropout=args.imle_configs.dropout,
                                     ensemble=args.sample_configs.ensemble,
                                     use_bn=args.imle_configs.batchnorm,
                                     deg_hist=args.ds_deg if hasattr(args, 'ds_deg') else None,
                                     upstream_model=args.imle_configs.upstream_model if hasattr(
                                         args.imle_configs, 'upstream_model') else None).to(device)
        else:
            raise NotImplementedError

    return model.to(device), emb_model
