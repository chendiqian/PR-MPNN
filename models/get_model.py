from data.const import DATASET_FEATURE_STAT_DICT
from models.downstream_models.ogb_mol_gnn import OGBGNN
from models.downstream_models.zinc_gin import ZINC_GIN
from models.upstream_models.linear_embed import LinearEmbed


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
                         num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'])
    else:
        raise NotImplementedError

    if args.imle_configs is not None:
        ensemble = 1 if not hasattr(args.sample_configs, 'ensemble') else args.sample_configs.ensemble
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
        else:
            raise NotImplementedError
    else:
        emb_model = None

    return model.to(device), emb_model.to(device) if emb_model is not None else None
