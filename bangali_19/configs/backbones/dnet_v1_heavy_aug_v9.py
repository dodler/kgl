config = {
    "arch": "multi-head",
    "backbone": "densenet161",
    "pretrained": True,
    "in-bn": False,
    'opt': 'radam',
    'loss_aggregate_fn': 'mean',
    'schedule': None,
    'T_0': 6,
    'lr': 1e-2,
    'train_aug': 'augmentations.compose.v0_heavy_norm',
    'valid_aug': 'augmentations.compose.v0_heavy_norm',
    'img_path': '/home/lyan/train/',
    # 'head': 'V2',
    'folds_path': '/home/lyan/Documents/kaggle/bangali_19/folds5.csv',
    'mixup': False,
    'mixup_alpha': 1.0,

}
