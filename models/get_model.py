import torch.nn
import torch.nn as nn
from functools import partial
from data.const import DATASET_FEATURE_STAT_DICT
from data.get_sampler import get_sampler
from models.downstream_models.gnn_duo import GNN_Duo
from models.downstream_models.gnn_halftransformer import GNN_HalfTransformer
from models.downstream_models.gnn_normal import GNN_Normal
from models.downstream_models.dynamic_rewire_gnn import (DynamicRewireGNN,
                                                         DecoupledDynamicRewireGNN,
                                                         DynamicRewireTransUpstreamGNN,
                                                         DecoupledDynamicRewireTransUpstreamGNN)
from models.my_encoder import FeatureEncoder, BondEncoder, QM9FeatureEncoder
from models.upstream_models.edge_candidate_selector import EdgeSelector
from models.upstream_models.edge_candidate_selector_2wl import EdgeSelector2WL
from models.upstream_models.transformer import Transformer
from models.downstream_models.qm9_gnn import QM9_Net
from training.construct import construct_from_edge_candidate, construct_from_attention_mat


def get_encoder(args, for_downstream):
    if args.dataset.lower() in ['zinc', 'alchemy', 'edge_wt_region_boundary', 'qm9', 'exp', 
                                'cexp', 'proteins', 'mutag', 'ptc_mr', 'nci1', 'nci109', 'csl']:
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

    if args.model.lower() in ['trans+gin', 'trans+gine']:
        # need to overwrite
        hidden = args.tf_hid_size

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
                                'ppgnqm9', 'exp', 'cexp', 'sym_skipcircles', 'ptc_mr', 'mutag', 'csl']:
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
            num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
            use_bn=args.bn,
            dropout=args.dropout,
            residual=args.residual,
            mlp_layers_intragraph=args.mlp_layers_intragraph,
            graph_pooling=args.graph_pooling)
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
                        in_features=DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
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
    elif args.model.lower().startswith('dynamic'):
        train_forward, val_forward, sampler_class = get_sampler(args.imle_configs,
                                                                args.sample_configs,
                                                                device)
        def make_intermediate_gnn():
            # downstream GNN
            return GNN_Duo(
                lambda data: data.x,
                edge_encoder,
                args.model.lower().split('_')[-1],  # gin or gine
                share_weights=False,
                include_org=args.sample_configs.include_original_graph,
                num_candidates=2 if args.sample_configs.separate and args.sample_configs.sample_k2 > 0 else 1,
                deg_hist=args.ds_deg if hasattr(args, 'ds_deg') else None,
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
        if args.model.lower().split('_')[1] == 'trans':
            # transformer upstream
            sampler = partial(construct_from_attention_mat,
                              samplek_dict={'add_k': args.sample_configs.sample_k,
                                            'del_k': args.sample_configs.sample_k2},
                              sample_policy='global_' + (
                                  'directed' if args.sample_configs.directed else 'undirected'),
                              directed_sampling=args.sample_configs.directed,
                              auxloss_dict=args.imle_configs.auxloss if hasattr(
                                  args.imle_configs,
                                  'auxloss') else None,
                              sampler_class=sampler_class,
                              train_forward=train_forward,
                              val_forward=val_forward,
                              weight_edges=args.imle_configs.weight_edges,
                              marginals_mask=args.imle_configs.marginals_mask,
                              device=device,
                              include_original_graph=args.sample_configs.include_original_graph,
                              in_place=args.sample_configs.in_place,
                              separate=args.sample_configs.separate,
                              num_layers=None,
                              rewire_layers=None,)
            if 'decoupled' in args.model.lower():
                model = DecoupledDynamicRewireTransUpstreamGNN(
                    sampler,
                    make_intermediate_gnn,
                    encoder,
                    hid_size=args.hid_size,
                    gnn_layer=args.num_convlayers,
                    num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                    dropout=args.dropout,
                    num_heads=args.num_heads,
                    ensemble=args.sample_configs.ensemble,
                    mlp_layers_intragraph=args.mlp_layers_intragraph,
                    graph_pooling=args.graph_pooling,
                    sample_alpha=args.sample_configs.sample_alpha if hasattr(
                        args.sample_configs,
                        'sample_alpha') else 1,
                    input_from_downstream=args.imle_configs.input_from_downstream if hasattr(args.imle_configs, 'input_from_downstream') else 0
                )
                surrogate_model = torch.nn.ModuleList([model.tfs, model.attns])
                emb_model = torch.nn.ModuleList([model.mlp, model.intermediate_gnns, model.atom_encoder])
            else:
                model = DynamicRewireTransUpstreamGNN(sampler,
                                                      make_intermediate_gnn,
                                                      encoder,
                                                      hid_size=args.hid_size,
                                                      num_heads=args.num_heads,
                                                      gnn_layer=args.num_convlayers,
                                                      num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                                                      dropout=args.dropout,
                                                      ensemble=args.sample_configs.ensemble,
                                                      mlp_layers_intragraph=args.mlp_layers_intragraph,
                                                      graph_pooling=args.graph_pooling,
                                                      sample_alpha=args.sample_configs.sample_alpha if hasattr(
                                                          args.sample_configs,
                                                          'sample_alpha') else 1, )
                surrogate_model = torch.nn.ModuleList([model.tfs, model.attns])
                emb_model = torch.nn.ModuleList([model.mlp, model.intermediate_gnns, model.atom_encoder])
        else:
            sampler = partial(construct_from_edge_candidate,
                              samplek_dict={'add_k': args.sample_configs.sample_k,
                                            'del_k': args.sample_configs.sample_k2},
                              sampler_class=sampler_class,
                              train_forward=train_forward,
                              val_forward=val_forward,
                              weight_edges=args.imle_configs.weight_edges,
                              marginals_mask=args.imle_configs.marginals_mask,
                              include_original_graph=args.sample_configs.include_original_graph,
                              separate=args.sample_configs.separate,
                              in_place=args.sample_configs.in_place,
                              directed_sampling=args.sample_configs.directed,
                              num_layers=None,
                              rewire_layers=None,
                              auxloss_dict=args.imle_configs.auxloss if hasattr(args.imle_configs,
                                                                       'auxloss') else None,
            )
            if 'decoupled' in args.model.lower():
                model = DecoupledDynamicRewireGNN(
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
                    graph_pooling=args.graph_pooling,
                    sample_alpha=args.sample_configs.sample_alpha if hasattr(args.sample_configs, 'sample_alpha') else 1,
                    input_from_downstream=args.imle_configs.input_from_downstream if hasattr(args.imle_configs, 'input_from_downstream') else 0,)
                surrogate_model = torch.nn.ModuleList([model.convs, model.bns, model.add_mlp_heads, model.del_mlp_heads])
                emb_model = torch.nn.ModuleList([model.mlp, model.intermediate_gnns, model.atom_encoder])
            else:
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
                    graph_pooling=args.graph_pooling,
                    sample_alpha=args.sample_configs.sample_alpha if hasattr(args.sample_configs, 'sample_alpha') else 1,)
                surrogate_model = torch.nn.ModuleList([model.convs, model.bns, model.add_mlp_heads, model.del_mlp_heads])
                emb_model = torch.nn.ModuleList([model.mlp, model.intermediate_gnns, model.atom_encoder])
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
        elif args.imle_configs.model == '2wl_edge_selector':
            emb_model = EdgeSelector2WL(DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'] ** 2 +
                                        DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'],
                                        args.imle_configs.emb_hid_size,
                                        args.sample_configs.ensemble,
                                        args.imle_configs.gnn_layer,
                                        args.imle_configs.mlp_layer,
                                        args.sample_configs.directed)
        else:
            raise NotImplementedError

    if emb_model is not None:
        emb_model = emb_model.to(device)
    if surrogate_model is not None:
        surrogate_model = surrogate_model.to(device)
    return model.to(device), emb_model, surrogate_model
