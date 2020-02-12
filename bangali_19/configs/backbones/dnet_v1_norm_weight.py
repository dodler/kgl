config = {
    "arch": "multi-head",
    "backbone": "densenet161",
    "pretrained": True,
    "in-bn": False,
    'opt': 'radam',
    'loss_aggregate_fn': 'weighted_sum',
    'loss_weights': [0.7, 0.1, 0.2],
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 6,
    'lr': 1e-3,
    'train_aug': 'augmentations.color.v2',
    'valid_aug': 'augmentations.color.v2',
}