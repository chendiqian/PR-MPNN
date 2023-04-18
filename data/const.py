from torch import nn


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

    'tree_2': {'node': 4, 'edge': 0, 'num_class': 4},
    'tree_3': {'node': 8, 'edge': 0, 'num_class': 8},
    'tree_4': {'node': 16, 'edge': 0, 'num_class': 16},
    'tree_5': {'node': 32, 'edge': 0, 'num_class': 32},
    'tree_6': {'node': 64, 'edge': 0, 'num_class': 64},
    'tree_7': {'node': 128, 'edge': 0, 'num_class': 128},
    'tree_8': {'node': 256, 'edge': 0, 'num_class': 256},
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
}

TASK_TYPE_DICT = {
    'zinc': 'regression',
    'zinc_full': 'regression',
    'alchemy': 'regression',
    'ogbg-molesol': 'rmse',
    'ogbg-molbace': 'rocauc',
    'ogbg-molhiv': 'rocauc',
    'ogbg-moltox21': 'rocauc',
    'qm9': 'regression',
    'exp': 'acc',
    'protein': 'acc',

    'tree_2': 'acc',
    'tree_3': 'acc',
    'tree_4': 'acc',
    'tree_5': 'acc',
    'tree_6': 'acc',
    'tree_7': 'acc',
    'tree_8': 'acc',
}

CRITERION_DICT = {
    'zinc': nn.L1Loss(),
    'zinc_full': nn.L1Loss(),
    'alchemy': nn.L1Loss(),
    'ogbg-molesol': nn.MSELoss(),
    'ogbg-molbace': nn.BCEWithLogitsLoss(),
    'ogbg-molhiv': nn.BCEWithLogitsLoss(),
    'ogbg-moltox21': nn.BCEWithLogitsLoss(),
    'qm9': nn.L1Loss(),
    'exp': nn.BCEWithLogitsLoss(),
    'protein': nn.BCEWithLogitsLoss(),

    'tree_2': nn.CrossEntropyLoss(),
    'tree_3': nn.CrossEntropyLoss(),
    'tree_4': nn.CrossEntropyLoss(),
    'tree_5': nn.CrossEntropyLoss(),
    'tree_6': nn.CrossEntropyLoss(),
    'tree_7': nn.CrossEntropyLoss(),
    'tree_8': nn.CrossEntropyLoss(),
}
