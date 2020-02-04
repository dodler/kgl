config = {
    "arch": "multi-head",
    "backbone": "densenet201",
    "pretrained": True,
    "in-bn": False,
    'opt': 'radam',
    'loss_aggregate_fn': 'mean',
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 6,
    'lr': 1e-3,
    'train_aug': 'augmentations.color.v2',
    'valid_aug': 'augmentations.geom.v0',
}