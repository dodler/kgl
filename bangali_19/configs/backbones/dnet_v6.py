config = {
    "arch": "multi-head",
    "backbone": "densenet161",
    "pretrained": True,
    "in-bn": True,
    'opt': 'sgd',
    'loss_aggregate_fn': 'mean',
    'schedule': 'reduce_lr_on_plateau',
    'T_0': 6,
    'lr': 1e-1,
    'train_aug': 'augmentations.v0',
    'valid_aug': 'augmentations.v0',
    'img_path': '/home/lyan/train/',
    # 'head': 'V2',
    'folds_path': '/home/lyan/Documents/kaggle/bangali_19/folds5.csv',
    'mixup': True,
    'mixup_alpha': 0.4,

}
