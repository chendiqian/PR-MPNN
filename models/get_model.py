import torch.nn as nn
from data.const import DATASET_FEATURE_STAT_DICT, NUM_CANDID_DICT
from models.downstream_models.gin_duo import GIN_Duo
from models.downstream_models.gin_halftransformer import GIN_HalfTransformer
from models.downstream_models.gin_normal import GIN_Normal
from models.downstream_models.leafcolor_gnn import LeafColorGraphModel
from models.downstream_models.tree_gnn import TreeGraphModel
from models.my_encoder import FeatureEncoder
from models.upstream_models.edge_candidate_selector import EdgeSelector
from models.upstream_models.transformer import Transformer


def get_model(args, device, *_args):
    if args.dataset.lower() in ['zinc', 'alchemy', 'edge_wt_region_boundary',]:
        type_encoder = 'linear'
    if args.dataset.lower().startswith('hetero'):
        type_encoder = 'linear'
    elif args.dataset.lower().startswith('tree'):
        type_encoder = 'bi_embedding'
    elif args.dataset.lower().startswith('leafcolor'):
        type_encoder = 'bi_embedding_cat'
    elif args.dataset.lower().startswith('peptides'):
        type_encoder = 'peptides'
    else:
        raise ValueError

    encoder = FeatureEncoder(
        dim_in=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
        hidden=args.hid_size,
        type_encoder=type_encoder,
        lap_encoder=args.lap if hasattr(args, 'lap') else None,
        rw_encoder=args.rwse if hasattr(args, 'rwse') else None)

    if args.model.lower() == 'gin_normal':
        model = GIN_Normal(
            encoder,
            in_features=args.hid_size,
            num_layers=args.num_convlayers,
            hidden=args.hid_size,
            num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
            use_bn=args.bn,
            dropout=args.dropout,
            residual=args.residual,
            mlp_layers_intragraph=args.mlp_layers_intragraph,
            graph_pooling=args.graph_pooling)
    elif args.model.lower().startswith('gin_duo'):
        model = GIN_Duo(encoder,
                        share_weights=args.model.lower().endswith('shared'),
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
    elif args.model.lower().endswith('trans+gin'):
        # need to overwrite this
        encoder = FeatureEncoder(
            dim_in=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
            hidden=args.tf_hid_size,
            type_encoder=type_encoder,
            lap_encoder=args.lap if hasattr(args, 'lap') else None,
            rw_encoder=args.rwse if hasattr(args, 'rwse') else None)
        model = GIN_HalfTransformer(
            encoder=encoder,
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
            use_spectral_norm=args.imle_configs.spectral_norm,
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
        encoder = FeatureEncoder(
            dim_in=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
            hidden=args.imle_configs.emb_hid_size,
            type_encoder=type_encoder,
            lap_encoder=args.imle_configs.lap if hasattr(args.imle_configs, 'lap') else None,
            rw_encoder=args.imle_configs.rwse if hasattr(args.imle_configs, 'rwse') else None)

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
            if args.dataset.lower() in ['zinc', 'alchemy', 'edge_wt_region_boundary']:
                edge_encoder = nn.Sequential(nn.Linear(DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'], args.imle_configs.emb_hid_size),
                                             nn.ReLU(),
                                             nn.Linear(args.imle_configs.emb_hid_size, args.imle_configs.emb_hid_size))
            else:
                edge_encoder = None
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
