config = {
    "arch": "multi-head",
    "backbone": "se_resnext101_32x4d",
    "pretrained": True,
    "in-bn": True,
    'opt': 'sgd',
    'loss_aggregate_fn': 'mean',
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 6,
    'lr': 0.1,
    'train_aug': 'augmentations.spatial.v2',
    'valid_aug': 'augmentations.geom.v0',
}
