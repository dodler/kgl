config = {
    "arch": "multi-head",
    "backbone": "efficientnet-b7",
    "pretrained": True,
    "in-bn": True,
    'opt': 'sgd',
    'loss_aggregate_fn': 'mean',
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 6,
    'lr': 0.1,
    'dropout': 0.2,
    'train_aug': 'augmentations.spatial.v2',
    'valid_aug': 'augmentations.geom.v0',
}
