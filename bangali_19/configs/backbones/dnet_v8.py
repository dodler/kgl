config = {
    "arch": "multi-head",
    "backbone": "densenet161",
    "pretrained": True,
    "in-bn": False,
    'opt': 'sgd',
    'loss_aggregate_fn': 'mean',
    'schedule': 'reduce_lr_on_plateau',
    'T_0': 6,
    'lr': 1e-1,
    'train_aug': 'augmentations.color.v3',
    'valid_aug': 'augmentations.color.v3',
    'img_path': '/home/lyan/train/',
    'folds_path': '/home/lyan/Documents/kaggle/bangali_19/folds5.csv',
    'mixup': True,
    'mixup_alpha': 0.4,

}
