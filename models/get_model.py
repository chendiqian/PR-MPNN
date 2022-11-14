from .ogb_mol_gnn import OGBGNN, OGBGNN_inner
from .emb_model import UpStream
from .zinc_gin import ZINC_GIN_Inner, ZINC_GIN_Outer
from data.const import DATASET_FEATURE_STAT_DICT


def get_model(args, device, *_args):
    if args.model.lower() == 'ogb_gin':
        model = OGBGNN(
            num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
            extra_dim=args.sample_configs.extra_dim,
            num_layer=args.num_convlayers,
            emb_dim=args.hid_size,
            gnn_type='gin',
            virtual_node=False,
            drop_ratio=args.dropout,
        ).to(device)

        inner_model = OGBGNN_inner(
            num_layer=args.sample_configs.inner_layer,
            emb_dim=args.hid_size,
            gnn_type='gin',
            virtual_node=False,
            drop_ratio=args.dropout,
            subgraph2node_aggr=args.sample_configs.subgraph2node_aggr,
        ).to(device)
    elif args.model.lower() == 'zinc_gin':
        model = ZINC_GIN_Outer(num_layers=args.num_convlayers,
                               hidden=args.hid_size,
                               num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                               extra_dim=args.sample_configs.extra_dim,).to(device)

        inner_model = ZINC_GIN_Inner(num_layers=args.sample_configs.inner_layer,
                                     hidden=args.hid_size,
                                     subgraph2node_aggr=args.sample_configs.subgraph2node_aggr,).to(device)
    else:
        raise NotImplementedError

    if args.imle_configs is not None:
        ensemble = 1 if not hasattr(args.sample_configs, 'ensemble') else args.sample_configs.ensemble

        emb_model = UpStream(hid_size=args.imle_configs.emb_hid_size,
                             num_layer=args.imle_configs.emb_num_layer,
                             dropout=args.imle_configs.dropout,
                             ensemble=ensemble).to(device)
    else:
        emb_model = None

    return model, emb_model, inner_model
