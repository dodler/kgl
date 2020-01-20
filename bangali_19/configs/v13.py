config = {
    "arch": "multi-head",
    "backbone": "se_resnext50_32x4d",
    "pretrained": True,
    "in-bn": True,
    'opt': 'sgd',
    'loss_aggregate_fn': 'mean',
    'schedule': 'cyclic',
    'base_lr': 1e-4,
    'max_lr': 0.1
}
