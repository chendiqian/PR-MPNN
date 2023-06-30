import torch.nn as nn
from functools import partial
from data.const import DATASET_FEATURE_STAT_DICT
from data.get_sampler import get_sampler
from models.downstream_models.gnn_duo import GNN_Duo
from models.downstream_models.gnn_halftransformer import GNN_HalfTransformer
from models.downstream_models.gnn_normal import GNN_Normal
from models.downstream_models.dynamic_rewire_gnn import DynamicRewireGNN
from models.my_encoder import FeatureEncoder, BondEncoder
from models.upstream_models.edge_candidate_selector import EdgeSelector
from models.upstream_models.transformer import Transformer
from models.downstream_models.qm9_gin import QM9_NetGIN
from training.construct import construct_from_edge_candidate


def get_encoder(args, for_downstream):
    if args.dataset.lower() in ['zinc', 'alchemy', 'edge_wt_region_boundary', 'qm9']:
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
    if args.dataset.lower() in ['zinc', 'alchemy', 'edge_wt_region_boundary', 'qm9']:
        edge_encoder = nn.Sequential(
            nn.Linear(DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'], edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden))
    elif args.dataset.lower().startswith('hetero') or args.dataset.lower().startswith('tree') or args.dataset.lower().startswith('leafcolor'):
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
                        num_candidates=2 if args.sample_configs.separate and args.sample_configs.sample_k2 > 0 else 1,
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
    elif args.model.lower().startswith('dynamic_edge_candidate'):
        train_forward, val_forward, sampler_class = get_sampler(args.imle_configs,
                                                                args.sample_configs,
                                                                device)
        def make_intermediate_gnn():
            return GNN_Duo(
                lambda data: data.x,
                edge_encoder,
                args.model.lower().split('_')[-1],  # gin or gine
                share_weights=False,
                include_org=args.sample_configs.include_original_graph,
                num_candidates=2 if args.sample_configs.separate and args.sample_configs.sample_k2 > 0 else 1,
                in_features=args.hid_size,
                num_layers=args.intermediate_gnn_layers,
                hidden=args.hid_size,
                num_classes=args.hid_size,
                use_bn=args.bn,
                dropout=args.dropout,
                residual=args.residual,
                mlp_layers_intragraph=args.mlp_layers_intragraph,
                mlp_layers_intergraph=args.mlp_layers_intergraph,
                graph_pooling=None,
                inter_graph_pooling=args.inter_graph_pooling)
        sampler = partial(construct_from_edge_candidate,
                          ensemble=args.sample_configs.ensemble,
                          samplek_dict={'add_k': args.sample_configs.sample_k,
                                        'del_k': args.sample_configs.sample_k2},
                          ensemble=args.sample_configs.ensemble,
                          sampler_class=sampler_class,
                          train_forward=train_forward,
                          val_forward=val_forward,
                          weight_edges=args.imle_configs.weight_edges,
                          marginals_mask=args.imle_configs.marginals_mask,
                          include_original_graph=args.sample_configs.include_original_graph,
                          negative_sample=args.imle_configs.negative_sample,
                          separate=args.sample_configs.separate,
                          in_place=args.sample_configs.in_place,
                          directed_sampling=args.sample_configs.directed,
                          num_layers=None,
                          rewire_layers=None,
                          auxloss_dict=args.imle_configs.auxloss if hasattr(args.imle_configs,
                                                                   'auxloss') else None)
        model = DynamicRewireGNN(
            sampler,
            make_intermediate_gnn=make_intermediate_gnn,
            encoder=encoder,
            edge_encoder=edge_encoder,
            hid_size=args.hid_size,
            gnn_type=args.model.lower().split('_')[-1],
            gnn_layer=args.num_convlayers,
            sample_mlp_layer=args.sample_mlp_layer,
            num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
            directed_sampling=args.sample_configs.directed,
            residual=args.residual,
            dropout=args.dropout,
            ensemble=args.sample_configs.ensemble,
            use_bn=args.bn,
            mlp_layers_intragraph=args.mlp_layers_intragraph,
            graph_pooling=args.graph_pooling)
    elif args.model == 'qm9_gin':
        model = QM9_NetGIN(num_features=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                           num_classes=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['num_class'],
                           emb_sizes=args.hid_size,
                           num_layers=args.num_convlayers,
                           drpt_prob=args.dropout,
                           graph_pooling=args.graph_pooling)
    else:
        raise NotImplementedError

    if args.imle_configs is not None and args.imle_configs.model is not None:
        # not shared with the downstream
        type_encoder, encoder, edge_encoder = get_encoder(args, for_downstream=False)
        if args.imle_configs.model == 'transformer':
            emb_model = Transformer(encoder=encoder,
                                    hidden=args.imle_configs.emb_hid_size,
                                    layers=args.imle_configs.tf_layer,
                                    num_heads=args.imle_configs.heads,
                                    ensemble=args.sample_configs.ensemble * (args.num_convlayers if args.sample_configs.per_layer else 1),
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
                                     ensemble=args.sample_configs.ensemble * (args.num_convlayers if args.sample_configs.per_layer else 1),
                                     use_bn=args.imle_configs.batchnorm)
        else:
            raise NotImplementedError
    else:
        emb_model = None

    return model.to(device), emb_model.to(device) if emb_model is not None else None
