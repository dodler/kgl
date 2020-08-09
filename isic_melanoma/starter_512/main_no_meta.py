import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import Compose, RandomSizedCrop, Normalize, Resize, RandomCrop, CenterCrop, RandomResizedCrop, \
    CLAHE, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, LambdaLR, CosineAnnealingWarmRestarts

from isic_melanoma.starter_512.data import MelanomaDataset
from isic_melanoma.starter_512.mel_models import MelModel, get_mel_model

from warmup_scheduler import GradualWarmupScheduler

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--warm_up_step', type=int, default=2000)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--model', type=str, default='b0')
parser.add_argument('--early_stop_patience', type=int, default=10, )
parser.add_argument('--sched', type=str, default=None, choices=['cyclic', 'cosine_annealing_warm_restarts', 'exp'])
parser.add_argument('--opt', type=str, default='sgd')


class MelanomaModule(pl.LightningModule):

    def __init__(self, args=None):
        super().__init__()

        self.opt = None
        self.sched = None

        if args is not None:
            for k in vars(args):
                self.hparams[k] = getattr(args, k)

        if args is not None:
            fold = args.fold
        else:
            fold = 0

        df = pd.read_csv('/home/lyan/Documents/kaggle/isic_melanoma/group_fold_train_512.csv')
        self.train_df = df[df.fold != fold].reset_index().drop('index', axis=1).copy()
        self.valid_df = df[df.fold == fold].reset_index().drop('index', axis=1).copy()
        self.path = '/var/ssd_2t_1/kaggle_isic/ds_512_2/512x512-dataset-melanoma/512x512-dataset-melanoma'

        if args is not None:
            self.batch_size = args.batch_size
        else:
            self.batch_size = 1

        self.model = get_mel_model(mdl=args.model, meta=False)

        self.train_aug = Compose([
            Resize(width=400, height=400, always_apply=True),
            RandomResizedCrop(width=384, height=384, scale=(0.5, 0.9), ratio=(0.5, 1), always_apply=True),
            CLAHE(),
            ShiftScaleRotate(),
            Normalize(always_apply=True),
            ToTensorV2(always_apply=True),
        ], p=0.3)

        self.valid_aug = Compose([
            Resize(width=384, height=384, always_apply=True),
            Normalize(always_apply=True),
            ToTensorV2(always_apply=True),
        ])
        self.val_y_hat = []
        self.val_y = []

    def forward(self, x):
        return self.model(x)

    def _roc_auc_score(self, y_hat, y):

        b = y.shape[0]
        with torch.no_grad():
            y_hat_ = torch.sigmoid(y_hat).cpu().numpy()
            y_ = y.cpu().numpy()

            s = int(np.sum(y_))
            if s == b or s == 0:
                return 0

            return roc_auc_score(
                y_true=y_,
                y_score=y_hat_
            )

    def get_lr(self):
        ''' Returns current learning rate for schedulers '''
        if self.opt is None:
            raise ValueError('No learning rate schedulers initialized')
        else:
            for pg in self.opt.param_groups:
                return pg['lr']

    def training_step(self, batch, batch_nb):
        (x, meta), y = batch
        y = y.float()
        x = x.float()
        y_hat = self.forward((x, meta))

        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        score = self._roc_auc_score(y_hat=y_hat, y=y)
        tensorboard_logs = {
            'train/_loss': loss,
            'train/_roc_auc': score,
            'train/_lr': self.get_lr(),
        }

        return {'loss': loss, 'log': tensorboard_logs, '_roc_auc': score}

    def validation_step(self, batch, batch_nb):
        (x, meta), y = batch

        x = x.float()
        y = y.float()

        y_hat = self.forward((x, meta))

        with torch.no_grad():
            self.val_y_hat.append(y_hat.cpu())
            self.val_y.append(y.cpu())

        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        score = self._roc_auc_score(y_hat=y_hat, y=y)
        return {'val_loss': loss, 'val_roc_auc': score, '_roc_auc': score}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        y_hat = torch.cat(self.val_y_hat)
        y = torch.cat(self.val_y)

        val_roc_auc = self._roc_auc_score(y_hat=y_hat, y=y)

        self.val_y_hat = []
        self.val_y = []

        tensorboard_logs = {'val/_loss': avg_loss, 'val/_roc_auc': val_roc_auc}

        if self.sched is not None:
            self.sched.step(self.current_epoch)

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs, 'val_roc_auc': val_roc_auc}

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):

        # if self.trainer.global_step < self.hparams.warm_up_step:
        #     lr_scale = min(10., 10 * float(self.trainer.global_step + 1) / float(self.hparams.warm_up_step))
        #     for pg in optimizer.param_groups:
        #         pg['lr'] = lr_scale * self.hparams.lr
        # else:
        # self.sched.step()

        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        if self.hparams['opt'] == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        elif self.hparams['opt'] == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)

        if self.hparams['sched'] == 'cyclic:':
            scheduler = CyclicLR(optimizer=opt, base_lr=self.hparams.lr / 500, max_lr=self.hparams.lr / 10)
        elif self.hparams['sched'] == 'cosine_annealing_warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer=opt,
                                                    T_0=2000,
                                                    eta_min=self.hparams.lr / 1000.0,
                                                    T_mult=1,
                                                    )
        elif self.hparams['sched'] == 'exp':
            scheduler_steplr = ExponentialLR(opt, gamma=0.95)
            scheduler = GradualWarmupScheduler(opt, multiplier=1, total_epoch=5,
                                               after_scheduler=scheduler_steplr)
        else:
            scheduler = None

        self.sched = scheduler
        self.opt = opt
        return opt

    def train_dataloader(self):
        train_ds = MelanomaDataset(df=self.train_df, path=self.path, tfm=self.train_aug,
                                   meta_features=MelanomaDataset.get_meta_features())
        train_loader = DataLoader(train_ds, shuffle=True, num_workers=6, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        valid_ds = MelanomaDataset(df=self.valid_df, path=self.path, tfm=self.valid_aug,
                                   meta_features=MelanomaDataset.get_meta_features())

        valid_loader = DataLoader(valid_ds, shuffle=False, num_workers=8, batch_size=self.batch_size)
        return valid_loader


if __name__ == '__main__':
    args = parser.parse_args()

    module = MelanomaModule(args=args)

    early_stop = EarlyStopping(monitor='val_roc_auc', verbose=True, patience=args.early_stop_patience, mode='max')
    trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, early_stop_callback=early_stop)
    trainer.fit(module)
