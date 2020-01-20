config = {
    "arch": "multi-head",
    "backbone": "se_resnext50_32x4d",
    "pretrained": True,
    "in-bn": True,
    'opt': 'sgd',
    'loss_aggregate_fn': 'mean',
    'schedule': 'cyclic'
}