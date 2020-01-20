config = {
    "arch": "multi-head",
    "backbone": "se_resnext50_32x4d",
    "pretrained": True,
    "in-bn": True,
    'opt': 'sgd',
    'loss_aggregate_fn': 'mean',
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 4,
    'lr': 0.1,
}
