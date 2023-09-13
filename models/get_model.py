from data.const import DATASET_FEATURE_STAT_DICT
from models.downstream_models.gnn_duo import GNN_Duo
from models.downstream_models.gnn_halftransformer import GNN_HalfTransformer
from models.downstream_models.gnn_normal import GNN_Normal
from models.my_encoder import FeatureEncoder, BondEncoder
from models.upstream_models.edge_candidate_selector import EdgeSelector
from models.upstream_models.transformer import Transformer


def get_encoder(args, for_downstream):
    type_encoder = 'atomencoder'

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

    edge_encoder = BondEncoder(edge_hidden)

    return encoder, edge_encoder


def get_model(args, device, *_args):
    model, emb_model, surrogate_model = None, None, None
    encoder, edge_encoder = get_encoder(args, for_downstream=True)
    if args.model.lower() in ['gin_normal', 'gine_normal', 'pna_normal', 'gcn_normal']:
        model = GNN_Normal(
            encoder,
            edge_encoder,
            args.model.lower().split('_')[0],  # gin or gine
            deg_hist=args.ds_deg if hasattr(args, 'ds_deg') else None,
            in_features=args.hid_size,
            num_layers=args.num_convlayers,
            hidden=args.hid_size,
            num_classes=args.hid_size,
            use_bn=args.bn,
            dropout=args.dropout,
            residual=args.residual,
            mlp_layers_intragraph=args.mlp_layers_intragraph)
    elif 'duo' in args.model.lower():
        share_weights = args.model.lower().endswith('shared')  # default False

        gnn_type = args.model.lower().split('_')[0]
        if gnn_type.startswith('qm9'):
            assert not (hasattr(args.imle_configs, 'rwse') or hasattr(args, 'rwse')
                        or hasattr(args.imle_configs, 'lap') or hasattr(args, 'lap')), "Need a new node encoder!"

        model = GNN_Duo(encoder,
                        edge_encoder,
                        gnn_type,
                        share_weights=share_weights,
                        include_org=args.sample_configs.include_original_graph,
                        num_candidates=2 if args.sample_configs.separate and args.sample_configs.sample_k2 > 0 else 1,
                        deg_hist=args.ds_deg if hasattr(args, 'ds_deg') else None,
                        num_layers=args.num_convlayers,
                        hidden=args.hid_size,
                        num_classes=args.hid_size,
                        use_bn=args.bn,
                        dropout=args.dropout,
                        residual=args.residual,
                        mlp_layers_intragraph=args.mlp_layers_intragraph,
                        mlp_layers_intergraph=args.mlp_layers_intergraph,
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
    else:
        raise NotImplementedError

    if args.imle_configs is not None and args.imle_configs.model is not None:
        # not shared with the downstream
        encoder, edge_encoder = get_encoder(args, for_downstream=False)
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
                                     use_deletion_head=True,
                                     directed_sampling=args.sample_configs.directed,
                                     dropout=args.imle_configs.dropout,
                                     ensemble=args.sample_configs.ensemble,
                                     use_bn=args.imle_configs.batchnorm,
                                     deg_hist=args.ds_deg if hasattr(args, 'ds_deg') else None,
                                     upstream_model=args.imle_configs.upstream_model if hasattr(
                                         args.imle_configs, 'upstream_model') else None, )
        else:
            raise NotImplementedError

    if emb_model is not None:
        emb_model = emb_model.to(device)
    if surrogate_model is not None:
        surrogate_model = surrogate_model.to(device)
    return model.to(device), emb_model, surrogate_model
