from torch import nn


DATASET_FEATURE_STAT_DICT = {
    'zinc': {'node': 21, 'edge': 4, 'num_class': 1},  # regression
    'zinc_full': {'node': 28, 'edge': 3, 'num_class': 1},
    'mutag': {'node': 7, 'edge': 4, 'num_class': 1},  # bin classification
    'alchemy': {'node': 6, 'edge': 4, 'num_class': 12},  # regression, but 12 labels

    'proteins': {'node': 3, 'edge': 0, 'num_class': 1},  # bin classification
    'ptc_mr': {'node': 18, 'edge': 4, 'num_class': 1},  # bin classification
    'nci1': {'node': 37, 'edge': 0, 'num_class': 1},  # bin classification
    'nci109': {'node': 38, 'edge': 0, 'num_class': 1},  # bin classification
    'imdb-m': {'node': 1, 'edge': 1, 'num_class': 3},  # classification
    'imdb-b': {'node': 1, 'edge': 1, 'num_class': 1},  # bin classification

    'ogbg-molesol': {'node': 9, 'edge': 3, 'num_class': 1},  # regression
    'ogbg-molbace': {'node': 9, 'edge': 3, 'num_class': 1},  # bin classification
    'ogbg-molhiv': {'node': 9, 'edge': 3, 'num_class': 1},  # regression
    'ogbg-moltox21': {'node': 9, 'edge': 3, 'num_class': 12},  # binary classification, but 12 tasks
    'qm9': {'node': 15, 'edge': 4, 'num_class': 1},  # regression, 13 labels, but we train 1 each split
    'ppgnqm9': {'node': 13, 'edge': 4, 'num_class': 1},  # regression, 13 labels, but we train 1 each split
    'exp': {'node': 2, 'edge': 0, 'num_class': 1},  # bin classification
    'csl': {'node': 1, 'edge': 1, 'num_class': 10},

    'sym_limits1': {'node': 4, 'edge': 0, 'num_class': 2},
    'sym_limits2': {'node': 4, 'edge': 0, 'num_class': 2},
    'sym_triangles': {'node': 1, 'edge': 0, 'num_class': 2},
    'sym_4cycles': {'node': 1, 'edge': 0, 'num_class': 2},
    'sym_skipcircles': {'node': 1, 'edge': 3, 'num_class': 10},
    'sym_lcc': {'node': 1, 'edge': 0, 'num_class': 3},

    'peptides-struct': {'node': 9, 'edge': 4, 'num_class': 11},  # regression, but 11 labels

    'pcqm': {'node': 9, 'edge': 3, 'num_class': 1},  # link pred

    'peptides-func': {'node': 9, 'edge': 4, 'num_class': 10},  # 10-way classification

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
    'leafcolor_4': {'node': 31, 'tree_depth': 4, 'n_leaf_labels': 2, 'num_class': 7},
    'leafcolor_5': {'node': 63, 'tree_depth': 5, 'n_leaf_labels': 2},
    'leafcolor_6': {'node': 127, 'tree_depth': 6, 'n_leaf_labels': 2},
    'leafcolor_7': {'node': 255, 'tree_depth': 7, 'n_leaf_labels': 2},
    'leafcolor_8': {'node': 511, 'tree_depth': 8, 'n_leaf_labels': 2},

    'hetero_cornell': {'node': 1703, 'edge': 0, 'num_class': 5},
    'hetero_texas': {'node': 1703, 'edge': 0, 'num_class': 5},
    'hetero_wisconsin': {'node': 1703, 'edge': 0, 'num_class': 5},
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

    'sym_limits1': 16,
    'sym_limits2': 16,
    'sym_triangles': 60,
    'sym_4cycles': 16,
    'sym_skipcircles': 41,
    'sym_lcc': 10,
}

TASK_TYPE_DICT = {
    'zinc': 'mae',
    'alchemy': 'mae',
    'proteins': 'acc',
    'mutag': 'acc',
    'ptc_mr': 'acc',
    'nci1': 'acc',
    'nci109': 'acc',
    'imdb-m': 'acc',
    'imdb-b': 'acc',
    'csl': 'acc',

    'peptides-struct': 'mae',
    'peptides-func': 'ap',
    'pcqm': 'mrr',
    'ogbg-molesol': 'rmse',
    'ogbg-molbace': 'rocauc',
    'ogbg-molhiv': 'rocauc',
    'ogbg-moltox21': 'rocauc',
    'qm9': 'mae',
    'ppgnqm9': 'mae',
    'exp': 'acc',
    'hetero_cornell': 'acc',
    'hetero_texas': 'acc',
    'hetero_wisconsin': 'acc',

    'tree_2': 'acc',
    'tree_3': 'acc',
    'tree_4': 'acc',
    'tree_5': 'acc',
    'tree_6': 'acc',
    'tree_7': 'acc',
    'tree_8': 'acc',

    'sym_limits1': 'acc',
    'sym_limits2': 'acc',
    'sym_triangles': 'acc',
    'sym_4cycles': 'acc',
    'sym_skipcircles': 'acc',
    'sym_lcc': 'acc',

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
    'proteins': nn.BCEWithLogitsLoss(),
    'mutag': nn.BCEWithLogitsLoss(),
    'ptc_mr': nn.BCEWithLogitsLoss(),
    'nci1': nn.BCEWithLogitsLoss(),
    'nci109': nn.BCEWithLogitsLoss(),
    'imdb-m': nn.CrossEntropyLoss(),
    'imdb-b': nn.BCEWithLogitsLoss(),
    'csl': nn.CrossEntropyLoss(),

    'peptides-struct': nn.L1Loss(),
    'peptides-func': nn.BCEWithLogitsLoss(),
    'ogbg-molesol': nn.MSELoss(),
    'ogbg-molbace': nn.BCEWithLogitsLoss(),
    'ogbg-molhiv': nn.BCEWithLogitsLoss(),
    'ogbg-moltox21': nn.BCEWithLogitsLoss(),
    'qm9': nn.MSELoss(),
    'ppgnqm9': nn.MSELoss(),
    'exp': nn.BCEWithLogitsLoss(),

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

    'sym_limits1': nn.CrossEntropyLoss(),
    'sym_limits2': nn.CrossEntropyLoss(),
    'sym_triangles': nn.CrossEntropyLoss(),
    'sym_4cycles': nn.CrossEntropyLoss(),
    'sym_skipcircles': nn.CrossEntropyLoss(),
    'sym_lcc': nn.CrossEntropyLoss(),

    'hetero_cornell': nn.CrossEntropyLoss(),
    'hetero_texas': nn.CrossEntropyLoss(),
    'hetero_wisconsin': nn.CrossEntropyLoss(),
}

TYPE_ENCODER = {
    'zinc': 'linear',
    'alchemy': 'linear',
    'qm9': 'linear',
    'exp': 'linear',
    'cexp': 'linear',
    'proteins': 'linear',
    'mutag': 'linear',
    'ptc_mr': 'linear',
    'nci1': 'linear',
    'nci109': 'linear',
    'csl': 'linear',
    'imdb-m': 'linear',
    'imdb-b': 'linear',

    'hetero_cornell': 'linear',
    'hetero_texas': 'linear',
    'hetero_wisconsin': 'linear',

    'sym_limits1': 'linear',
    'sym_limits2': 'linear',
    'sym_triangles': 'linear',
    'sym_4cycles': 'linear',
    'sym_skipcircles': 'linear',
    'sym_lcc': 'linear',

    'tree_2': 'bi_embedding',
    'tree_3': 'bi_embedding',
    'tree_4': 'bi_embedding',
    'tree_5': 'bi_embedding',
    'tree_6': 'bi_embedding',
    'tree_7': 'bi_embedding',
    'tree_8': 'bi_embedding',

    'leafcolor_2': 'bi_embedding_cat',
    'leafcolor_3': 'bi_embedding_cat',
    'leafcolor_4': 'bi_embedding_cat',
    'leafcolor_5': 'bi_embedding_cat',
    'leafcolor_6': 'bi_embedding_cat',
    'leafcolor_7': 'bi_embedding_cat',
    'leafcolor_8': 'bi_embedding_cat',

    'peptides-struct': 'atomencoder',
    'peptides-func': 'atomencoder',
    'ogbg-molesol': 'atomencoder',
    'ogbg-molbace': 'atomencoder',
    'ogbg-molhiv': 'atomencoder',
    'ogbg-moltox21': 'atomencoder',
}
