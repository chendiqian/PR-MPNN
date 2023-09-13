from torch import nn


DATASET_FEATURE_STAT_DICT = {
    'pcqm-contact': {'node': 9, 'edge': 3, 'num_class': 1},  # link pred
}

TASK_TYPE_DICT = {
    'pcqm-contact': 'mrr',
}

CRITERION_DICT = {
    'pcqm-contact': nn.BCEWithLogitsLoss(),
}