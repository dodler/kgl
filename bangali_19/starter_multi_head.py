import argparse

import torch
from catalyst.dl import CriterionCallback, AccuracyCallback
from catalyst.dl.callbacks import EarlyStoppingCallback
from catalyst.dl.callbacks.criterion import CriterionAggregatorCallback
from catalyst.dl.runner import SupervisedRunner
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from bangali_19.beng_eff_net import BengEffNetClassifier
from bangali_19.beng_utils import bengali_ds_from_folds

parser = argparse.ArgumentParser(description='Understanding cloud training')

parser.add_argument('--lr',
                    default=1e-4,
                    type=float,
                    help='learning rate')
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--comment', type=str, default=None)
parser.add_argument('--swa', action='store_true')
parser.add_argument('--model', type=str, choices=['densenet121', 'densenet169', 'densenet201',
                                                  'densenet161', 'dpn68', 'dpn68b',
                                                  'dpn92', 'dpn98', 'dpn107', 'dpn131',
                                                  'inceptionresnetv2', 'resnet101', 'resnet152',
                                                  'se_resnet101', 'se_resnet152',
                                                  'se_resnext50_32x4d', 'se_resnext101_32x4d',
                                                  'senet154', 'se_resnet50', 'resnet50', 'resnet34',
                                                  'efficientnet-b0', 'efficientnet-b1',
                                                  'efficientnet-b2', 'efficientnet-b3',
                                                  'efficientnet-b4', 'efficientnet-b5'],
                    default='efficientnet-b0')
parser.add_argument('--fp16', action='store_true')
args = parser.parse_args()

model = BengEffNetClassifier()

num_workers = 10
bs = args.batch_size

train_dataset, valid_dataset = bengali_ds_from_folds(folds_path='/home/lyan/Documents/kaggle/bangali_19/folds.csv')

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

num_epochs = 40
logdir = "/var/data/bengali" + str(args.fold) + '_model_' + str(args.model)+'_comment_'+args.comment

optimizer = AdamW(params=model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)


criterion = {
    "h1": torch.nn.CrossEntropyLoss(),
    "h2": torch.nn.CrossEntropyLoss(),
    "h3": torch.nn.CrossEntropyLoss(),
}

runner = SupervisedRunner(input_key='features', output_key=["h1_logits", "h2_logits", 'h3_logits'])


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
        loss_aggregate_fn="sum"  # or "mean"
    ),
    AccuracyCallback(input_key='h1_targets', output_key='h1_logits', prefix='acc_h1_'),
    AccuracyCallback(input_key='h2_targets', output_key='h2_logits', prefix='acc_h2_'),
    AccuracyCallback(input_key='h3_targets', output_key='h3_logits', prefix='acc_h3_'),
    EarlyStoppingCallback(patience=10, min_delta=0.001)
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