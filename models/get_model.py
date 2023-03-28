import torch.nn

from data.const import DATASET_FEATURE_STAT_DICT
from models.downstream_models.ogb_mol_gnn import OGBGNN
from models.downstream_models.zinc_gin import ZINC_GIN
from models.upstream_models.linear_embed import LinearEmbed
from models.upstream_models.transformer import FeatureEncoder, Transformer
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
        model = ZINC_GIN(in_features=DATASET_FEATURE_STAT_DICT['zinc']['node'],
                         num_layers=args.num_convlayers,
                         hidden=args.hid_size,
                         num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                         mlp_layers_intragraph=args.mlp_layers_intragraph,
                         mlp_layers_intergraph=args.mlp_layers_intergraph)
    else:
        raise NotImplementedError

    if args.imle_configs is not None:
        ensemble = 1 if not hasattr(args.sample_configs, 'ensemble') else args.sample_configs.ensemble
        spectral_norm = True if hasattr(args.imle_configs, 'spectral_norm') and args.imle_configs.spectral_norm else False
        if args.imle_configs.model.startswith('lin'):
            emb_model = LinearEmbed(
                tuple_type=args.imle_configs.model.split('_')[-1],
                heads=args.imle_configs.heads if hasattr(args.imle_configs, 'heads') else 1,
                in_features=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                edge_features=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'],
                hid_size=args.imle_configs.emb_hid_size,
                gnn_layer=args.imle_configs.gnn_layer,
                mlp_layer=args.imle_configs.mlp_layer,
                dropout=args.imle_configs.dropout,
                emb_edge=args.imle_configs.emb_edge,
                emb_spd=args.imle_configs.emb_spd,
                emb_ppr=args.imle_configs.emb_ppr,
                ensemble=ensemble,
                use_bn=args.imle_configs.bn,
                use_ogb_encoder=args.dataset.lower().startswith('ogb')
            )
        elif args.imle_configs.model == 'transformer':
            if args.dataset.lower() in ['zinc']:
                type_encoder = 'linear'
            else:
                raise ValueError
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
                                   mlp_dropout=args.imle_configs.mlp_dropout)
        else:
            raise NotImplementedError
    else:
        emb_model = None

    return model.to(device), emb_model.to(device) if emb_model is not None else None
