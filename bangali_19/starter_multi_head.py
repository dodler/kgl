import argparse

import torch
from catalyst.dl import CriterionCallback, AccuracyCallback
from catalyst.dl.callbacks import EarlyStoppingCallback
from catalyst.dl.callbacks.criterion import CriterionAggregatorCallback
from catalyst.dl.runner import SupervisedRunner
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR, CyclicLR
from torch.utils.data import DataLoader

from bangali_19.augmentations import get_augmentation
from bangali_19.beng_eff_net import BengEffNetClassifier
from bangali_19.beng_resnets import BengResnet
from bangali_19.beng_utils import bengali_ds_from_folds, make_scheduler_from_config, get_dict_value_or_default
from bangali_19.configs import get_config

parser = argparse.ArgumentParser(description='Understanding cloud training')

parser.add_argument('--lr',
                    default=1e-4,
                    type=float,
                    help='learning rate')
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--comment', type=str, default='')
parser.add_argument('--config', type=str, required=True, default=None)
parser.add_argument('--fp16', action='store_true')
args = parser.parse_args()

config = get_config(name=args.config)

if config['arch'] == 'multi-head':
    if 'efficientnet' in config['backbone']:
        model = BengEffNetClassifier(
            name=config['backbone'],
            pretrained=config['pretrained'],
            input_bn=config['in-bn']
        )
    elif 'resnet' in config['backbone'] or 'resnext' in config['backbone']:
        model = BengResnet(
            name=config['backbone'],
            pretrained=config['pretrained'],
            input_bn=config['in-bn']
        )
    else:
        raise Exception('backbone ' + config['backbone'] + ' is not supported')
else:
    raise Exception(config['arch'] + ' is not supported')

num_workers = 6
bs = args.batch_size

train_aug = get_dict_value_or_default(config, 'train_aug', 'v0')
valid_aug = get_dict_value_or_default(config, 'valid_aug', 'v0')
train_aug, _ = get_augmentation(train_aug)
_, valid_aug = get_augmentation(valid_aug)

train_dataset, valid_dataset = bengali_ds_from_folds(
    folds_path='/home/lyan/Documents/kaggle/bangali_19/folds.csv',
    train_aug=train_aug,
    valid_aug=valid_aug,
)

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

num_epochs = 40
logdir = "/var/data/bengali" + str(args.fold) + '_config_' + str(args.config) + '_comment_' + args.comment

lr = get_dict_value_or_default(dict_=config, key='lr', default_value=args.lr)

if config['opt'] == 'adamw':
    optimizer = AdamW(params=model.parameters(), lr=lr)
elif config['opt'] == 'adam':
    optimizer = Adam(params=model.parameters(), lr=lr)
elif config['opt'] == 'sgd':
    optimizer = SGD(params=model.parameters(), lr=lr, momentum=0.9, nesterov=True)
elif config['opt'] == 'rmsprop':
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=lr)
else:
    raise Exception(config['opt'] + ' is not supported')

scheduler = make_scheduler_from_config(optimizer=optimizer, config=config)

criterion = {
    "h1": torch.nn.CrossEntropyLoss(),
    "h2": torch.nn.CrossEntropyLoss(),
    "h3": torch.nn.CrossEntropyLoss(),
}

runner = SupervisedRunner(input_key='features', output_key=["h1_logits", "h2_logits", 'h3_logits'])

early_stop_epochs = get_dict_value_or_default(dict_=config, key='early_stop_epochs', default_value=10)

callbacks = [
    CriterionCallback(
        input_key="h1_targets",
        output_key="h1_logits",
        prefix="loss_h1",
        criterion_key="h1"
    ),
    CriterionCallback(
        input_key="h2_targets",
        output_key="h2_logits",
        prefix="loss_h2",
        criterion_key="h2"
    ),
    CriterionCallback(
        input_key="h3_targets",
        output_key="h3_logits",
        prefix="loss_h3",
        criterion_key="h3"
    ),
    CriterionAggregatorCallback(
        prefix="loss",
        loss_keys=["loss_h1", "loss_h2", 'loss_h3'],
        loss_aggregate_fn=config['loss_aggregate_fn']  # or "mean"
    ),
    AccuracyCallback(input_key='h1_targets', output_key='h1_logits', prefix='acc_h1_'),
    AccuracyCallback(input_key='h2_targets', output_key='h2_logits', prefix='acc_h2_'),
    AccuracyCallback(input_key='h3_targets', output_key='h3_logits', prefix='acc_h3_'),
    EarlyStoppingCallback(patience=early_stop_epochs, min_delta=0.001)
]

runner.train(
    fp16=args.fp16,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=callbacks,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)
