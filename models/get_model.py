import torch.nn

from data.const import DATASET_FEATURE_STAT_DICT
from models.downstream_models.ogb_mol_gnn import OGBGNN
from models.downstream_models.zinc_gin import ZINC_GIN
from models.downstream_models.alchemy_gin import AL_GIN
from models.downstream_models.tree_gnn import TreeGraphModel
from models.downstream_models.leafcolor_gnn import LeafColorGraphModel
from models.downstream_models.zinc_halftransformer import ZINC_HalfTransformer
from models.downstream_models.alchemy_halftransformer import AL_HalfTransformer
from models.upstream_models.linear_embed import LinearEmbed
from models.upstream_models.transformer import Transformer
from models.my_encoder import FeatureEncoder
from models.upstream_models.graphormer import BiasEncoder, NodeEncoder, Graphormer


def get_model(args, device, *_args):
    if args.model.lower() == 'ogb_gin':
        model = OGBGNN(
            num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
            num_layer=args.num_convlayers,
            emb_dim=args.hid_size,
            gnn_type='gin',
            virtual_node=False,
            drop_ratio=args.dropout,
        )
    elif args.model.lower() == 'zinc_gin':
        if hasattr(args, 'lap') or hasattr(args, 'rwse'):
            # we encode the lap and rwse to the downstream model
            encoder = FeatureEncoder(
                dim_in=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                hidden=args.input_feature,
                type_encoder='linear',
                lap_encoder=args.lap if hasattr(args, 'lap') else None,
                rw_encoder=args.rwse if hasattr(args, 'rwse') else None)
            input_feature = args.input_feature
        else:
            encoder = None
            input_feature = DATASET_FEATURE_STAT_DICT['zinc']['node']

        model = ZINC_GIN(
            encoder=encoder,
            ensemble=args.sample_configs.ensemble + int(args.sample_configs.include_original_graph),
            in_features=input_feature,
            num_layers=args.num_convlayers,
            hidden=args.hid_size,
            num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
            mlp_layers_intragraph=args.mlp_layers_intragraph,
            mlp_layers_intergraph=args.mlp_layers_intergraph,
            inter_graph_pooling=args.inter_graph_pooling)
    elif args.model.lower() == 'alchemy_gin':
        if hasattr(args, 'lap') or hasattr(args, 'rwse'):
            # we encode the lap and rwse to the downstream model
            encoder = FeatureEncoder(
                dim_in=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                hidden=args.hid_size,
                type_encoder='linear',
                lap_encoder=args.lap if hasattr(args, 'lap') else None,
                rw_encoder=args.rwse if hasattr(args, 'rwse') else None)
            input_feature = args.hid_size
        else:
            encoder = None
            input_feature = DATASET_FEATURE_STAT_DICT['alchemy']['node']

        model = AL_GIN(
            encoder=encoder,
            ensemble=args.sample_configs.ensemble + int(args.sample_configs.include_original_graph),
            in_features=input_feature,
            num_layers=args.num_convlayers,
            hidden=args.hid_size,
            num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
            mlp_layers_intragraph=args.mlp_layers_intragraph,
            mlp_layers_intergraph=args.mlp_layers_intergraph,
            inter_graph_pooling=args.inter_graph_pooling)
    elif args.model.lower().endswith('trans+gin'):
        encoder = FeatureEncoder(
            dim_in=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
            hidden=args.tf_hid_size,
            type_encoder='linear',
            lap_encoder=args.lap if hasattr(args, 'lap') else None,
            rw_encoder=args.rwse if hasattr(args, 'rwse') else None)
        if args.model.lower().split('_')[0] == 'zinc':
            model_class = ZINC_HalfTransformer
        elif args.model.lower().split('_')[0] == 'alchemy':
            model_class = AL_HalfTransformer
        else:
            raise ValueError
        model = model_class(
            encoder=encoder,
            head=args.tf_head,
            gnn_in_features=args.tf_hid_size,
            num_layers=args.num_convlayers,
            tf_layers=args.tf_layers,
            hidden=args.hid_size,
            tf_hidden=args.tf_hid_size,
            dropout=args.tf_dropout,
            attn_dropout=args.attn_dropout,
            num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
            mlp_layers_intragraph=args.mlp_layers_intragraph,
            layer_norm=False,
            batch_norm=True,
            use_spectral_norm=True,
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
        spectral_norm = True if hasattr(args.imle_configs, 'spectral_norm') and args.imle_configs.spectral_norm else False
        if args.dataset.lower() in ['zinc', 'alchemy']:
            type_encoder = 'linear'
        elif args.dataset.lower().startswith('tree'):
            type_encoder = 'bi_embedding'
        elif args.dataset.lower().startswith('leafcolor'):
            type_encoder = 'bi_embedding_cat'
        else:
            raise ValueError
        if args.imle_configs.model.startswith('lin'):
            encoder = FeatureEncoder(
                dim_in=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                hidden=args.imle_configs.emb_hid_size,
                type_encoder=type_encoder,
                lap_encoder=args.imle_configs.lap if hasattr(args.imle_configs, 'lap') else None,
                rw_encoder=args.imle_configs.rwse if hasattr(args.imle_configs, 'rwse') else None)
            emb_model = LinearEmbed(
                encoder=encoder,
                tuple_type=args.imle_configs.model.split('_')[-1],
                heads=args.imle_configs.heads if hasattr(args.imle_configs, 'heads') else 1,
                edge_features=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'],
                hid_size=args.imle_configs.emb_hid_size,
                gnn_layer=args.imle_configs.gnn_layer,
                mlp_layer=args.imle_configs.mlp_layer,
                dropout=args.imle_configs.dropout,
                emb_edge=args.imle_configs.emb_edge,
                emb_spd=args.imle_configs.emb_spd,
                emb_ppr=args.imle_configs.emb_ppr,
                ensemble=args.sample_configs.ensemble,
                use_bn=args.imle_configs.bn,
            )
        elif args.imle_configs.model == 'transformer':
            encoder = FeatureEncoder(dim_in=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                                     hidden=args.imle_configs.emb_hid_size,
                                     type_encoder=type_encoder,
                                     lap_encoder=args.imle_configs.lap if hasattr(args.imle_configs, 'lap') else None,
                                     rw_encoder=args.imle_configs.rwse if hasattr(args.imle_configs, 'rwse') else None)
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
                                    use_spectral_norm=spectral_norm)
        elif args.imle_configs.model == 'graphormer':
            if args.dataset.lower() in ['zinc']:
                encoder = torch.nn.Linear(DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                                          args.imle_configs.emb_hid_size)
            else:
                raise ValueError
            bias_encoder = BiasEncoder(num_heads=args.imle_configs.heads,
                                       num_spatial_types=args.imle_configs.attenbias.num_spatial_types,
                                       num_edge_types=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'],)
            node_encoder = NodeEncoder(embed_dim=args.imle_configs.emb_hid_size,
                                       num_in_degree=args.imle_configs.attenbias.num_in_degrees,
                                       num_out_degree=args.imle_configs.attenbias.num_in_degrees,
                                       input_dropout=args.imle_configs.input_dropout,)
            emb_model = Graphormer(encoder=encoder,
                                   bias_conder=bias_encoder,
                                   node_encoder=node_encoder,
                                   hidden=args.imle_configs.emb_hid_size,
                                   layers=args.imle_configs.tf_layer,
                                   num_heads=args.imle_configs.heads,
                                   ensemble=args.sample_configs.ensemble,
                                   dropout=args.imle_configs.dropout,
                                   attn_dropout=args.imle_configs.attn_dropout,
                                   mlp_dropout=args.imle_configs.mlp_dropout,
                                   use_spectral_norm=spectral_norm)
        else:
            raise NotImplementedError
    else:
        emb_model = None

    return model.to(device), emb_model.to(device) if emb_model is not None else None
