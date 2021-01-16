import argparse

from snapmix import *
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from cassava.radam import RAdam

from cassava.snapmix import SnapMixLoss, snapmix
from grad_cent import AdamW_GCC2
from data import CassavaDs
from cassava.model_snap import CassavaModel
from aug import get_aug


def get_or_default(d, key, default_value):
    if key in d:
        return d[key]
    else:
        return default_value


SNAPMIX_ALPHA = 5.0
SNAPMIX_PCT = 1.0
device = 0


class CassavaModule(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = CassavaModel(cfg=cfg)
        self.crit = nn.CrossEntropyLoss()
        self.snap_mix_crit = SnapMixLoss()
        trn_params = cfg['train_params']
        self.fold = get_or_default(trn_params, 'fold', 0)
        self.batch_size = get_or_default(trn_params, 'batch_size', 16)
        self.num_workers = get_or_default(trn_params, 'num_workers', 2)
        self.csv_path = get_or_default(cfg, 'csv_path', 'input/train_folds_merged.csv')
        self.trn_path = get_or_default(cfg, 'image_path', 'input/train_merged/')
        self.aug_type = get_or_default(cfg, 'aug', '0')
        print('using fold', self.fold, 'trn path', self.trn_path)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        rand = np.random.rand()
        if rand > (1.0 - SNAPMIX_PCT):
            X, ya, yb, lam_a, lam_b = snapmix(x, y, SNAPMIX_ALPHA, self.model)
            outputs, _ = self.model(X)
            loss = self.snap_mix_crit(self.crit, outputs, ya, yb, lam_a, lam_b)
        else:
            outputs, _ = self.model(x)
            loss = self.crit(outputs, y)

        acc = accuracy_score(y.detach().cpu().numpy(), np.argmax(outputs.detach().cpu().numpy(), axis=1))

        self.log('trn/_loss', loss)
        self.log('trn/_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.forward(x)
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
        elif self.cfg['optimizer'] == 'radam':
            optimizer = RAdam(self.model.parameters(), lr=lr)
        else:
            raise Exception('optimizer {} not supported'.format(self.cfg['optimizer']))

        self.opt = optimizer

        if self.cfg['scheduler']['type'] == 'CosineAnnealingWarmRestarts':
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
            raise Exception('scheduler {} not supported')
        sched = {'scheduler': sched, 'name': 'adam+{}'.format(self.cfg['scheduler']['type'])}
        return [optimizer], [sched]

    def train_dataloader(self):

        trn_aug = get_aug(atype=self.aug_type, size=self.cfg['img_size'])

        train = pd.read_csv(self.csv_path)
        trn_ds = CassavaDs(df=train[train.fold != self.fold].reset_index().drop('index', axis=1).drop('fold', axis=1),
                           aug=trn_aug, path=self.trn_path)

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
    args = parser.parse_args()

    cfg = benedict.from_yaml(args.config)
    module = CassavaModule(cfg=cfg)

    early_stop = EarlyStopping(monitor='val/avg_acc', verbose=True, patience=10, mode='max')
    logger = TensorBoardLogger("lightning_logs", name=args.config)
    lrm = LearningRateMonitor()
    mdl_ckpt = ModelCheckpoint(monitor='val/avg_acc', save_top_k=3, mode='max')
    precision = get_or_default(cfg['train_params'], key='precision', default_value=32)
    clip_grad = get_or_default(cfg['train_params'], key='clip_grad', default_value=0.0)
    trainer = pl.Trainer(gpus=1, max_epochs=60, callbacks=[early_stop, lrm, mdl_ckpt], logger=logger,
                         precision=precision, gradient_clip_val=clip_grad)

    trainer.fit(module)
