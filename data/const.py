from torch import nn

from data.data_utils import weighted_cross_entropy


DATASET_FEATURE_STAT_DICT = {
    'zinc': {'node': 21, 'edge': 4, 'num_class': 1},  # regression
    'zinc_full': {'node': 28, 'edge': 3, 'num_class': 1},
    'mutag': {'node': 7, 'edge': 4, 'num_class': 1},  # bin classification
    'alchemy': {'node': 6, 'edge': 4, 'num_class': 12},  # regression, but 12 labels
    'ogbg-molesol': {'node': 9, 'edge': 3, 'num_class': 1},  # regression
    'ogbg-molbace': {'node': 9, 'edge': 3, 'num_class': 1},  # bin classification
    'ogbg-molhiv': {'node': 9, 'edge': 3, 'num_class': 1},  # regression
    'ogbg-moltox21': {'node': 9, 'edge': 3, 'num_class': 12},  # binary classification, but 12 tasks
    'qm9': {'node': 11, 'edge': 5, 'num_class': 12},  # regression, 12 labels
    'exp': {'node': 1, 'edge': 0, 'num_class': 1},  # bin classification
    'protein': {'node': 3, 'edge': 0, 'num_class': 1},  # bin classification

    'peptides-struct': {'node': 9, 'edge': 4, 'num_class': 11},  # regression, but 11 labels

    # VOC superpixels:
    'edge_wt_region_boundary': {'node': 14, 'edge': 2, 'num_class': 21},

    'tree_2': {'node': 4, 'edge': 0, 'num_class': 4},
    'tree_3': {'node': 8, 'edge': 0, 'num_class': 8},
    'tree_4': {'node': 16, 'edge': 0, 'num_class': 16},
    'tree_5': {'node': 32, 'edge': 0, 'num_class': 32},
    'tree_6': {'node': 64, 'edge': 0, 'num_class': 64},
    'tree_7': {'node': 128, 'edge': 0, 'num_class': 128},
    'tree_8': {'node': 256, 'edge': 0, 'num_class': 256},

    # maybe we can make the labels const and add the n_labels here
    'leafcolor_2': {'node': 7, 'tree_depth': 2, 'n_leaf_labels': 2},
    'leafcolor_3': {'node': 15, 'tree_depth': 3, 'n_leaf_labels': 2},
    'leafcolor_4': {'node': 31, 'tree_depth': 4, 'n_leaf_labels': 2},
    'leafcolor_5': {'node': 63, 'tree_depth': 5, 'n_leaf_labels': 2},
    'leafcolor_6': {'node': 127, 'tree_depth': 6, 'n_leaf_labels': 2},
    'leafcolor_7': {'node': 255, 'tree_depth': 7, 'n_leaf_labels': 2},
    'leafcolor_8': {'node': 511, 'tree_depth': 8, 'n_leaf_labels': 2},
}

MAX_NUM_NODE_DICT = {
    'zinc': 37,
    'zinc_full': 38,
    'ogbg-molesol': 55,
    'ogbg-molbace': 97,
    'ogbg-molhiv': 222,
    'tree_2': 7,
    'tree_3': 15,
    'tree_4': 31,
    'tree_5': 63,
    'tree_6': 127,
    'tree_7': 255,
    'tree_8': 511,

    'leafcolor_2': 7,
    'leafcolor_3': 15,
    'leafcolor_4': 31,
    'leafcolor_5': 63,
    'leafcolor_6': 127,
    'leafcolor_7': 255,
    'leafcolor_8': 511,
}

TASK_TYPE_DICT = {
    'zinc': 'mae',
    # 'zinc_full': 'regression',
    'alchemy': 'mae',
    'peptides-struct': 'mae',
    'ogbg-molesol': 'rmse',
    'ogbg-molbace': 'rocauc',
    'ogbg-molhiv': 'rocauc',
    'ogbg-moltox21': 'rocauc',
    # 'qm9': 'regression',
    'exp': 'acc',
    'protein': 'acc',

    # VOC superpixels
    'edge_wt_region_boundary': 'f1',

    'tree_2': 'acc',
    'tree_3': 'acc',
    'tree_4': 'acc',
    'tree_5': 'acc',
    'tree_6': 'acc',
    'tree_7': 'acc',
    'tree_8': 'acc',

    'leafcolor_2': 'acc',
    'leafcolor_3': 'acc',
    'leafcolor_4': 'acc',
    'leafcolor_5': 'acc',
    'leafcolor_6': 'acc',
    'leafcolor_7': 'acc',
    'leafcolor_8': 'acc',
}

CRITERION_DICT = {
    'zinc': nn.L1Loss(),
    'zinc_full': nn.L1Loss(),
    'alchemy': nn.L1Loss(),
    'peptides-struct': nn.L1Loss(),
    'ogbg-molesol': nn.MSELoss(),
    'ogbg-molbace': nn.BCEWithLogitsLoss(),
    'ogbg-molhiv': nn.BCEWithLogitsLoss(),
    'ogbg-moltox21': nn.BCEWithLogitsLoss(),
    'qm9': nn.L1Loss(),
    'exp': nn.BCEWithLogitsLoss(),
    'protein': nn.BCEWithLogitsLoss(),

    'edge_wt_region_boundary': weighted_cross_entropy,

    'tree_2': nn.CrossEntropyLoss(),
    'tree_3': nn.CrossEntropyLoss(),
    'tree_4': nn.CrossEntropyLoss(),
    'tree_5': nn.CrossEntropyLoss(),
    'tree_6': nn.CrossEntropyLoss(),
    'tree_7': nn.CrossEntropyLoss(),
    'tree_8': nn.CrossEntropyLoss(),

    'leafcolor_2': nn.CrossEntropyLoss(),
    'leafcolor_3': nn.CrossEntropyLoss(),
    'leafcolor_4': nn.CrossEntropyLoss(),
    'leafcolor_5': nn.CrossEntropyLoss(),
    'leafcolor_6': nn.CrossEntropyLoss(),
    'leafcolor_7': nn.CrossEntropyLoss(),
    'leafcolor_8': nn.CrossEntropyLoss(),
}
