config = {
    "arch": "multi-head",
    "backbone": "se_resnext50_32x4d",
    "pretrained": True,
    "in-bn": True,
    'opt': 'sgd',
    'loss_aggregate_fn': 'weighted_sum',
    'loss_weights': [0.2, 0.5, 0.5],
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 6,
    'lr': 0.1,
    'train_aug': 'augmentations.color.v1',
    'valid_aug': 'augmentations.geom.v0',
}
