config = {
    "arch": "multi-head",
    "backbone": "efficientnet-b4",
    "pretrained": True,
    "in-bn": False,
    'opt': 'over9000',
    'loss_aggregate_fn': 'mean',
    'schedule': 'cosine_annealing_warm_restarts',
    'T_0': 8,
    'lr': 1e-2,
    'train_aug': 'augmentations.compose.v2_heavy_norm',
    'valid_aug': 'augmentations.compose.v2_heavy_norm',
    'img_path': '/home/lyan/train/',
    'head': 'V3',
    'iafoss_head': False,
    'folds_path': '/home/lyan/Documents/kaggle/bangali_19/folds5.csv',
    'mixup': False,
    'early_stop': False,
    'epochs': 100,
    'mixup_alpha': 1.0,

}
