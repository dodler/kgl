import sys

from catalyst.contrib.nn import OneCycleLRWithWarmup

from bangali_19.beng_fastai_dnet import Dnet_1ch

sys.path.append('/home/lyan/Documents/over9000')
from over9000 import Over9000

import argparse

import torch
from catalyst.dl import CriterionCallback
from catalyst.dl.callbacks import EarlyStoppingCallback, CriterionAggregatorCallback, OptimizerCallback
from catalyst.dl.runner import SupervisedRunner
from torch.utils.data import DataLoader

from bangali_19.augmentations import get_augmentation
from bangali_19.beng_score import score_callback
from bangali_19.beng_utils import bengali_ds_from_folds, get_dict_value_or_default
from bangali_19.configs import get_config
from catalyst_contrib import MixupCallback

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

model = Dnet_1ch()

num_workers = 8
bs = args.batch_size

channel_num = 1

train_aug = 'augmentations.compose.v2_heavy_norm'
valid_aug = 'augmentations.compose.v2_heavy_norm'
train_aug, _ = get_augmentation(train_aug)
_, valid_aug = get_augmentation(valid_aug)

img_path = '/home/lyan/train/'
folds_path = '/home/lyan/Documents/kaggle/bangali_19/folds5.csv'

train_dataset, valid_dataset = bengali_ds_from_folds(
    img_path=img_path,
    folds_path=folds_path,
    train_aug=train_aug,
    valid_aug=valid_aug,
    channel_num=channel_num,
)

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

num_epochs = 50
logdir = "/var/data/bengali" + str(args.fold) + '_config_' + str(args.config) + '_comment_' + args.comment

lr = 1e-2

optimizer = Over9000(params=model.parameters(), lr=lr)

scheduler = OneCycleLRWithWarmup(
    optimizer,
    num_steps=num_epochs,
    lr_range=(0.2e-2, 1e-2),
    warmup_steps=2,
    momentum_range=(1e-3, 0.1e-1)
)

criterion = {
    "h1": torch.nn.CrossEntropyLoss(),
    "h2": torch.nn.CrossEntropyLoss(),
    "h3": torch.nn.CrossEntropyLoss(),
}

runner = SupervisedRunner(input_key='features', output_key=["h1_logits", "h2_logits", 'h3_logits'])

early_stop_epochs = get_dict_value_or_default(dict_=config, key='early_stop_epochs', default_value=30)

loss_agg_fn = get_dict_value_or_default(config, 'loss_aggregate_fn', 'mean')
if loss_agg_fn == 'mean' or loss_agg_fn == 'sum':
    crit_agg = CriterionAggregatorCallback(
        prefix="loss",
        loss_keys=["loss_h1", "loss_h2", 'loss_h3'],
        loss_aggregate_fn=config['loss_aggregate_fn']
    )
elif loss_agg_fn == 'weighted_sum':
    weights = get_dict_value_or_default(config, 'weights', [0.3, 0.3, 0.3])
    crit_agg = CriterionAggregatorCallback(
        prefix="loss",
        loss_keys={"loss_h1": weights[0], "loss_h2": weights[1], 'loss_h3': weights[2]},
        loss_aggregate_fn=config['loss_aggregate_fn'],
    )

callbacks = []

mixup = get_dict_value_or_default(config, key='mixup', default_value=False)
mixup_alpha = get_dict_value_or_default(config, key='mixup_alpha', default_value=0.3)

if mixup:
    callbacks.extend([
        MixupCallback(crit_key='h1', input_key='h1_targets', output_key='h1_logits', alpha=mixup_alpha,
                      on_train_only=False),
    ])
else:
    callbacks.extend([
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
        crit_agg,
    ])

callbacks.extend([
    score_callback,
    EarlyStoppingCallback(metric='weight_recall', patience=early_stop_epochs, min_delta=0.001)
])

callbacks.append(
    OptimizerCallback(grad_clip_params={'params': 1.0}),
)

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
    verbose=True,
)
