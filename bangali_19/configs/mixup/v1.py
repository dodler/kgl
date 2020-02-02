config = {
    "arch": "multi-head",
    "backbone": "se_resnext50_32x4d",
    "pretrained": True,
    "in-bn": True,
    'opt': 'radam',
    'loss_aggregate_fn': 'mean',
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 6,
    'lr': 1e-3,
    'train_aug': 'augmentations.spatial.v2',
    'valid_aug': 'augmentations.geom.v0',
    'mixup':True,
    'mixup_alpha':0.3,
}
