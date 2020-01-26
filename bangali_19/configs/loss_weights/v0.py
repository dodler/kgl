config = {
    "arch": "multi-head",
    "backbone": "se_resnext50_32x4d",
    "pretrained": True,
    "in-bn": True,
    'opt': 'rmsprop',
    'loss_aggregate_fn': 'weighted_sum',
    'loss_weight': [0.8, 0.1, 0.1],
}
