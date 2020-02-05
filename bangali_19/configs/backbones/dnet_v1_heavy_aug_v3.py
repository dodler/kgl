config = {
    "arch": "multi-head",
    "backbone": "densenet161",
    "pretrained": True,
    "in-bn": True,
    'opt': 'radam',
    'loss_aggregate_fn': 'weighted_sum',
    'loss_weights': [0.7, 0.1, 0.2],
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 6,
    'lr': 1e-3,
    'train_aug': 'augmentations.compose.v0_heavy',
    'valid_aug': 'augmentations.geom.v0',
}
