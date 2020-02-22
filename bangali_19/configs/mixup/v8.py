config = {
    "arch": "multi-head",
    "backbone": "densenet121",
    "pretrained": True,
    "in-bn": False,
    'opt': 'radam',
    'loss_aggregate_fn': 'mean',
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 6,
    'lr': 4e-3,
    'early_stop_epochs': 50,
    'epochs': 150,
    'train_aug': 'augmentations.v1',
    'valid_aug': 'augmentations.v1',
    'img_path': '/home/lyan/train/',
    'mixup': True,
    'mixup_alpha': 0.4,
}

