config = {
    'from': 'mixup.v2',
    "arch": "multi-head",
    "backbone": "resnext101_32x8d_wsl",
    "pretrained": True,
    "in-bn": True,
    'opt': 'radam',
    'loss_aggregate_fn': 'mean',
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 6,
    'lr': 1e-3,
    'early_stop_epochs': 50,
    'train_aug': 'augmentations.spatial.v2',
    'valid_aug': 'augmentations.spatial.v2',
    'mixup': True,
    'mixup_alpha': 0.2,
    'img_path': '/home/lyan/train/',
}
