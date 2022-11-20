from data.const import DATASET_FEATURE_STAT_DICT
from models.downstream_models.planetoid_gcn import PlanetoidGCN
from models.downstream_models.ogb_mol_gnn import OGBGNN, OGBGNN_inner
from models.downstream_models.zinc_gin import ZINC_GIN_Inner, ZINC_GIN_Outer
from models.upstream_models.gcn_embed import GCN_Embed
from models.upstream_models.linear_embed import LinearEmbed
from models.upstream_models.transformer import Transformer
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
        model = BindModel(inner_model, outer_model).to(device)
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
        model = BindModel(inner_model, outer_model).to(device)
    elif args.model.lower() == 'planetoid_gcn':
        model = PlanetoidGCN(num_convlayers=args.num_convlayers,
                             in_features=DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                             hid=args.hid_size,
                             num_classes=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                             dropout=args.dropout).to(device)
    else:
        raise NotImplementedError

    if args.imle_configs is not None:
        ensemble = 1 if not hasattr(args.sample_configs, 'ensemble') else args.sample_configs.ensemble

        if not hasattr(args.imle_configs, 'model') or args.imle_configs.model == 'simple':
            emb_model = LinearEmbed(
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
        elif args.imle_configs.model == 'planetoid_gcn':
            emb_model = GCN_Embed(num_convlayers=args.imle_configs.emb_num_layer,
                                  in_features=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                                  hid=args.imle_configs.emb_hid_size,
                                  num_classes=args.sample_configs.ensemble,
                                  dropout=args.imle_configs.dropout).to(device)
        else:
            raise NotImplementedError
    else:
        emb_model = None

    return model, emb_model
