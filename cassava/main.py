import argparse
from collections import Counter

import albumentations as alb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from benedict import benedict
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_toolbelt.losses import FocalLoss
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from cassava.aug import get_aug
from cassava.data import CassavaDs
from cassava.focal_cosine_loss import FocalCosineLoss
from cassava.ldam import LDAMLoss
from cassava.model import CassavaModel
from cassava.smoothed_loss import SmoothCrossEntropyLoss
from cassava.taylor_smooth import TaylorCrossEntropyLoss
from opts.grad_cent import AdamW_GCC2
from seed import seed_everything

from cutmix.utils import CutMixCrossEntropyLoss
from cutmix.cutmix import CutMix

SEED = 2020
seed_everything(SEED)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_or_default(d, key, default_value):
    if key in d:
        return d[key]
    else:
        return default_value


class CassavaModule(pl.LightningModule):

    def __init__(self, cfg, fold=0):
        super().__init__()
        self.cfg = cfg
        self.model = CassavaModel(cfg=cfg)
        trn_params = cfg['train_params']
        self.fold = fold
        self.batch_size = get_or_default(trn_params, 'batch_size', 16)
        self.num_workers = get_or_default(trn_params, 'num_workers', 2)
        self.aug_type = get_or_default(cfg, 'aug', '0')
        self.csv_path = get_or_default(cfg, 'csv_path', 'input/train_folds.csv')
        self.trn_path = get_or_default(cfg, 'image_path', 'input/train_merged')
        self.mixup = get_or_default(cfg, 'mixup', False)
        self.do_cutmix = False

        if 'crit' not in cfg:
            self.crit = nn.CrossEntropyLoss()
        elif cfg['crit'] == 'focal':
            self.crit = FocalLoss()
        elif cfg['crit'] == 'smooth':
            self.crit = SmoothCrossEntropyLoss(smoothing=0.05)
        elif cfg['crit'] == 'cutmix':
            self.crit = CutMixCrossEntropyLoss(True)
            self.do_cutmix = True
        elif self.cfg['crit'] == 'focal_cosine':
            self.crit = FocalCosineLoss()
        elif self.cfg['crit'] == 'ldam':
            labels_list = list(Counter(pd.read_csv(self.csv_path).label.values).values())
            self.crit = LDAMLoss(labels_list)
        elif self.cfg['crit'] == 'taylor+smooth':
            self.crit = TaylorCrossEntropyLoss()
        else:
            raise Exception('criterion not specified')
        print('mixup', self.mixup)

        print('using fold', self.fold)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.mixup:
            x, y_a, y_b, lam = mixup_data(x, y)

        y_hat = self.forward(x)

        if self.mixup:
            loss = mixup_criterion(self.crit, y_hat, y_a, y_b, lam)
        elif self.do_cutmix:
            loss = self.crit(y_hat, y)
        else:
            loss = self.crit(y_hat, y)

        self.log('trn/_loss', loss)
        if not self.do_cutmix:
            acc = accuracy_score(y.detach().cpu().numpy(), np.argmax(y_hat.detach().cpu().numpy(), axis=1))
            self.log('trn/_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy_score(y.detach().cpu().numpy(), np.argmax(y_hat.detach().cpu().numpy(), axis=1))

        self.log('val/_loss', loss)
        self.log('val/_acc', acc, prog_bar=True)

        return loss, acc

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x[0] for x in outputs]).mean()
        avg_acc = np.array([x[1] for x in outputs]).mean()
        self.log('val/avg_loss', avg_loss)
        self.log('val/avg_acc', avg_acc)

    def configure_optimizers(self):

        lr = float(self.cfg['train_params']['lr'])

        if self.cfg['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif self.cfg['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif self.cfg['optimizer'] == 'adamw_gcc2':
            optimizer = AdamW_GCC2(self.model.parameters(), lr=lr)
        else:
            raise Exception('optimizer {} not supported'.format(self.cfg['optimizer']))

        self.opt = optimizer

        if self.cfg['scheduler']['type'] == 'none':
            sched = None
        elif self.cfg['scheduler']['type'] == 'CosineAnnealingWarmRestarts':
            T_mult = self.cfg['scheduler']['T_mult']
            T_0 = self.cfg['scheduler']['T_0']
            eta_min = float(self.cfg['scheduler']['eta_min'])
            sched = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=-1)
        elif self.cfg['scheduler']['type'] == 'OneCycleLR':
            max_lr = float(self.cfg['scheduler']['max_lr'])
            steps_per_epoch = cfg['scheduler']['steps_per_epoch']
            epochs = cfg['scheduler']['epochs']
            sched = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
        else:
            raise Exception('scheduler {} not supported'.format(self.cfg['scheduler']['type']))
        if sched is not None:
            sched = {'scheduler': sched, 'name': format(self.cfg['scheduler']['type'])}

        if sched is not None:
            return [optimizer], [sched]
        else:
            return optimizer

    def train_dataloader(self):

        trn_aug = get_aug(atype=self.aug_type, size=self.cfg['img_size'])

        train = pd.read_csv(self.csv_path)
        trn_ds = CassavaDs(df=train[train.fold != self.fold].reset_index().drop('index', axis=1).drop('fold', axis=1),
                           aug=trn_aug, path=self.trn_path)
        if self.do_cutmix:
            trn_ds = CutMix(trn_ds, num_class=5, beta=1.0, prob=0.5, num_mix=2)

        trn_dl = torch.utils.data.DataLoader(trn_ds, shuffle=True,
                                             batch_size=self.batch_size,
                                             num_workers=self.num_workers)
        return trn_dl

    def val_dataloader(self):
        val_aug = alb.Compose([
            alb.Resize(self.cfg['img_size'], self.cfg['img_size'], p=1),
            alb.Normalize(p=1),
            ToTensorV2(p=1),
        ])

        train = pd.read_csv(self.csv_path)

        val_ds = CassavaDs(df=train[train.fold == self.fold].reset_index().drop('index', axis=1).drop('fold', axis=1),
                           aug=val_aug, path=self.trn_path)
        val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False,
                                             batch_size=self.batch_size,
                                             num_workers=self.num_workers)
        return val_dl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, required=False)
    parser.add_argument('--fold', type=int, required=False, default=0)
    args = parser.parse_args()

    cfg = benedict.from_yaml(args.config)
    module = CassavaModule(cfg=cfg, fold=args.fold)

    early_stop = EarlyStopping(monitor='val/avg_acc', verbose=True, patience=20, mode='max')
    output_name = args.config + '_fold_' + str(args.fold)
    logger = TensorBoardLogger("lightning_logs", name=output_name)
    lrm = LearningRateMonitor()
    mdl_ckpt = ModelCheckpoint(monitor='val/avg_acc', save_top_k=1, mode='max')
    precision = get_or_default(cfg['train_params'], 'precision', 32)
    grad_clip = float(get_or_default(cfg['train_params'], 'grad_clip', 0))
    epochs = int(get_or_default(cfg['train_params'], 'epochs', 80))
    trainer = pl.Trainer(gpus=1, max_epochs=epochs, callbacks=[early_stop, lrm, mdl_ckpt],
                         logger=logger, precision=precision, gradient_clip_val=grad_clip)

    trainer.fit(module)
