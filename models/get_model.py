from data.const import DATASET_FEATURE_STAT_DICT
from .emb_model import UpStream
from .ogb_mol_gnn import OGBGNN, OGBGNN_inner
from .transformer import Transformer
from .zinc_gin import ZINC_GIN_Inner, ZINC_GIN_Outer
from .bind_model import BindModel


def get_model(args, device, *_args):
    if args.model.lower() == 'ogb_gin':
        outer_model = OGBGNN(
            num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
            extra_dim=args.sample_configs.extra_dim,
            num_layer=args.num_convlayers,
            emb_dim=args.hid_size,
            gnn_type='gin',
            virtual_node=False,
            drop_ratio=args.dropout,
        )
        inner_model = OGBGNN_inner(
            num_layer=args.sample_configs.inner_layer,
            emb_dim=args.hid_size,
            gnn_type='gin',
            virtual_node=False,
            drop_ratio=args.dropout,
            subgraph2node_aggr=args.sample_configs.subgraph2node_aggr,
        )
    elif args.model.lower() == 'zinc_gin':
        outer_model = ZINC_GIN_Outer(in_features=DATASET_FEATURE_STAT_DICT['zinc']['node'],
                               edge_features=DATASET_FEATURE_STAT_DICT['zinc']['edge'],
                               num_layers=args.num_convlayers,
                               hidden=args.hid_size,
                               num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                               extra_dim=args.sample_configs.extra_dim, )
        inner_model = ZINC_GIN_Inner(in_features=DATASET_FEATURE_STAT_DICT['zinc']['node'],
                                     edge_features=DATASET_FEATURE_STAT_DICT['zinc']['edge'],
                                     num_layers=args.sample_configs.inner_layer,
                                     hidden=args.hid_size,
                                     subgraph2node_aggr=args.sample_configs.subgraph2node_aggr, )
    else:
        raise NotImplementedError

    model = BindModel(inner_model, outer_model).to(device)

    if args.imle_configs is not None:
        ensemble = 1 if not hasattr(args.sample_configs, 'ensemble') else args.sample_configs.ensemble

        if not hasattr(args.imle_configs, 'model') or args.imle_configs.model == 'simple':
            emb_model = UpStream(
                in_features=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                edge_features=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'],
                hid_size=args.imle_configs.emb_hid_size,
                num_layer=args.imle_configs.emb_num_layer,
                dropout=args.imle_configs.dropout,
                ensemble=ensemble,
                use_ogb_encoder=args.dataset.lower().startswith('ogb')
            ).to(device)
        elif args.imle_configs.model == 'trans':
            emb_model = Transformer(num_layers=args.imle_configs.emb_num_layer,
                                    kq_dim=args.imle_configs.kq_dim,
                                    v_dim=args.imle_configs.emb_hid_size,
                                    edge_mlp_hid=args.imle_configs.edge_mlp_hid,
                                    ensemble=ensemble).to(device)
        else:
            raise NotImplementedError
    else:
        emb_model = None

    return model, emb_model
