config = {
    "arch": "multi-head",
    "backbone": "resnext101_32x8d_wsl",
    "pretrained": True,
    "in-bn": True,
    'opt': 'radam',
    'loss_aggregate_fn': 'mean',
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 6,
    'lr': 1e-3,
    'dropout': 0.4,
    'train_aug': 'augmentations.spatial.v2_heavy',
    'valid_aug': 'augmentations.geom.v0',
}