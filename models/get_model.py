import torch.nn as nn
from data.const import DATASET_FEATURE_STAT_DICT, NUM_CANDID_DICT
from models.downstream_models.gnn_duo import GNN_Duo
from models.downstream_models.gnn_halftransformer import GNN_HalfTransformer
from models.downstream_models.gnn_normal import GNN_Normal
from models.downstream_models.leafcolor_gnn import LeafColorGraphModel
from models.downstream_models.tree_gnn import TreeGraphModel
from models.my_encoder import FeatureEncoder, BondEncoder
from models.upstream_models.edge_candidate_selector import EdgeSelector
from models.upstream_models.transformer import Transformer


def get_encoder(args, for_downstream):
    if args.dataset.lower() in ['zinc', 'alchemy', 'edge_wt_region_boundary',]:
        type_encoder = 'linear'
    elif args.dataset.lower().startswith('hetero'):
        type_encoder = 'linear'
    elif args.dataset.lower().startswith('tree'):
        type_encoder = 'bi_embedding'
    elif args.dataset.lower().startswith('leafcolor'):
        type_encoder = 'bi_embedding_cat'
    elif args.dataset.lower().startswith('peptides'):
        type_encoder = 'peptides'
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

    if args.model.lower() in ['trans+gin', 'trans+gine']:
        # need to overwrite
        hidden = args.tf_hid_size

    encoder = FeatureEncoder(
        dim_in=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
        hidden=hidden,
        type_encoder=type_encoder,
        lap_encoder=lap,
        rw_encoder=rwse)
    if args.dataset.lower() in ['zinc', 'alchemy', 'edge_wt_region_boundary']:
        edge_encoder = nn.Sequential(
            nn.Linear(DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'], edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden))
    elif args.dataset.lower().startswith('hetero') or args.model.lower().startswith('tree') or args.model.lower().startswith('leafcolor'):
        edge_encoder = None
    elif args.dataset.lower().startswith('peptides'):
        edge_encoder = BondEncoder(edge_hidden)
    else:
        raise NotImplementedError("we need torch.nn.Embedding, to be implemented")

    return type_encoder, encoder, edge_encoder


def get_model(args, device, *_args):
    type_encoder, encoder, edge_encoder = get_encoder(args, for_downstream=True)
    if args.model.lower() in ['gin_normal', 'gine_normal']:
        model = GNN_Normal(
            encoder,
            edge_encoder,
            args.model.lower().split('_')[0],  # gin or gine
            in_features=args.hid_size,
            num_layers=args.num_convlayers,
            hidden=args.hid_size,
            num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
            use_bn=args.bn,
            dropout=args.dropout,
            residual=args.residual,
            mlp_layers_intragraph=args.mlp_layers_intragraph,
            graph_pooling=args.graph_pooling)
    elif 'duo' in args.model.lower():
        share_weights = args.model.lower().endswith('shared')  # default False
        model = GNN_Duo(encoder,
                        edge_encoder,
                        args.model.lower().split('_')[0],  # gin or gine
                        share_weights=share_weights,
                        include_org=args.sample_configs.include_original_graph,
                        num_candidates=NUM_CANDID_DICT[args.sample_configs.sample_policy],
                        in_features=args.hid_size,
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
    elif args.model.lower() in ['trans+gin', 'trans+gine']:
        model = GNN_HalfTransformer(
            encoder=encoder,
            edge_encoder=edge_encoder,
            base_gnn=args.model.lower().split('+')[-1],
            head=args.tf_head,
            gnn_in_features=args.tf_hid_size,
            num_layers=args.num_convlayers,
            tf_layers=args.tf_layers,
            hidden=args.hid_size,
            tf_hidden=args.tf_hid_size,
            tf_dropout=args.tf_dropout,
            attn_dropout=args.attn_dropout,
            num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
            mlp_layers_intragraph=args.mlp_layers_intragraph,
            layer_norm=False,
            batch_norm=True,
            use_spectral_norm=True,
            graph_pooling=args.graph_pooling,
        )
    elif args.model.lower().startswith('tree'):
        model = TreeGraphModel(gnn_type=args.model.lower().split('_')[1],
                               num_layers=args.num_convlayers,
                               dim0=DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                               h_dim=args.hid_size,
                               out_dim=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                               last_layer_fully_adjacent=False,
                               unroll=False,
                               layer_norm=False,
                               use_activation=False,
                               use_residual=False)
    elif args.model.lower().startswith('leafcolor'):
        model = LeafColorGraphModel(gnn_type=args.model.lower().split('_')[1],
                                    num_layers=args.num_convlayers,
                                    tree_depth=DATASET_FEATURE_STAT_DICT[args.dataset]['tree_depth'],
                                    n_leaf_labels=DATASET_FEATURE_STAT_DICT[args.dataset]['n_leaf_labels'],
                                    h_dim=args.hid_size,
                                    out_dim=args['num_classes'],
                                    last_layer_fully_adjacent=False,
                                    unroll=False,
                                    layer_norm=False,
                                    use_activation=False,
                                    use_residual=False)
    else:
        raise NotImplementedError

    if args.imle_configs is not None:
        # not shared with the downstream
        type_encoder, encoder, edge_encoder = get_encoder(args, for_downstream=False)
        if args.imle_configs.model == 'transformer':
            emb_model = Transformer(encoder=encoder,
                                    hidden=args.imle_configs.emb_hid_size,
                                    layers=args.imle_configs.tf_layer,
                                    num_heads=args.imle_configs.heads,
                                    ensemble=args.sample_configs.ensemble,
                                    act='relu',
                                    dropout=args.imle_configs.dropout,
                                    attn_dropout=args.imle_configs.attn_dropout,
                                    layer_norm=args.imle_configs.layernorm,
                                    batch_norm=args.imle_configs.batchnorm,
                                    use_spectral_norm=args.imle_configs.spectral_norm)
        elif args.imle_configs.model == 'edge_selector':
            emb_model = EdgeSelector(encoder,
                                     edge_encoder,
                                     in_dim=args.imle_configs.emb_hid_size,
                                     hid_size=args.imle_configs.emb_hid_size,
                                     gnn_layer=args.imle_configs.gnn_layer,
                                     mlp_layer=args.imle_configs.mlp_layer,
                                     use_deletion_head=True if args.sample_configs.sample_policy in ['edge_candid_bi', 'edge_candid_seq'] else False,
                                     dropout=args.imle_configs.dropout,
                                     ensemble=args.sample_configs.ensemble,
                                     use_bn=args.imle_configs.batchnorm)
        else:
            raise NotImplementedError
    else:
        emb_model = None

    return model.to(device), emb_model.to(device) if emb_model is not None else None
