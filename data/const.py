import torch
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

    'cora': {'node': 1433, 'edge': 0, 'num_class': 7},
    'pubmed': {'node': 500, 'edge': 0, 'num_class': 3},
}

MAX_NUM_NODE_DICT = {
    'zinc': 37,
    'zinc_full': 38,
    'ogbg-molesol': 55,
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

    'cora': 'acc',
    'pubmed': 'acc',
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

    'cora': lambda pred, y: nn.CrossEntropyLoss()(pred, y.to(torch.long)),
    'pubmed': lambda pred, y: nn.CrossEntropyLoss()(pred, y.to(torch.long)),
}
