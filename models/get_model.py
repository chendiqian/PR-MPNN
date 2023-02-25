from data.const import DATASET_FEATURE_STAT_DICT
from models.downstream_models.planetoid_gcn import PlanetoidGCN
from models.downstream_models.planetoid_gin import PlanetoidGIN
from models.downstream_models.ogb_mol_gnn import OGBGNN, OGBGNN_inner
from models.downstream_models.zinc_gin import ZINC_GIN_Inner, ZINC_GIN_Outer
from models.upstream_models.gcn_embed import GCN_Embed
from models.upstream_models.linear_embed import LinearEmbed
from models.upstream_models.fwl2_embed import Fwl2Embed
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
            drop_ratio=0.,
            subgraph2node_aggr=args.sample_configs.subgraph2node_aggr,
        )
        model = BindModel(inner_model, outer_model)
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
        model = BindModel(inner_model, outer_model)
    elif args.model.lower() == 'planetoid_gcn':
        model = PlanetoidGCN(num_convlayers=args.num_convlayers,
                             in_features=DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                             hid=args.hid_size,
                             num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                             dropout=args.dropout,
                             aggr=args.sample_configs.subgraph2node_aggr)
    elif args.model.lower() == 'planetoid_gin':
        model = PlanetoidGIN(num_convlayers=args.num_convlayers,
                             in_features=DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                             hid=args.hid_size,
                             num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                             dropout=args.dropout,
                             aggr=args.sample_configs.subgraph2node_aggr)
    else:
        raise NotImplementedError

    if args.imle_configs is not None:
        ensemble = 1 if not hasattr(args.sample_configs, 'ensemble') else args.sample_configs.ensemble

        if args.imle_configs.model.startswith('lin'):
            emb_model = LinearEmbed(
                tuple_type=args.imle_configs.model.split('_')[-1],
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
        elif args.imle_configs.model == 'planetoid_gcn':
            emb_model = GCN_Embed(num_convlayers=args.imle_configs.emb_num_layer,
                                  in_features=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                                  hid=args.imle_configs.emb_hid_size,
                                  num_classes=args.sample_configs.ensemble,
                                  dropout=args.imle_configs.dropout)
        elif args.imle_configs.model == 'fwl':
            emb_model = Fwl2Embed(
                in_features=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                edge_features=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'],
                hid_size=args.imle_configs.emb_hid_size,
                fwl_layer=args.imle_configs.gnn_layer,
                mlp_layer=args.imle_configs.mlp_layer,
                dropout=args.imle_configs.dropout,
                emb_edge=args.imle_configs.emb_edge,
                emb_spd=args.imle_configs.emb_spd,
                emb_ppr=args.imle_configs.emb_ppr,
                ensemble=ensemble,
                use_norm=args.imle_configs.bn,
                use_ogb_encoder=args.dataset.lower().startswith('ogb'))
        else:
            raise NotImplementedError
    else:
        emb_model = None

    return model.to(device), emb_model.to(device) if emb_model is not None else None
