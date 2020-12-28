import argparse

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
from cassava.model import CassavaModel
from cassava.rev_grad import RevGrad
from cassava.smoothed_loss import SmoothCrossEntropyLoss
from grad_cent import AdamW_GCC2
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

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = CassavaModel(cfg=cfg)
        trn_params = cfg['train_params']
        self.fold = get_or_default(trn_params, 'fold', 0)
        self.batch_size = get_or_default(trn_params, 'batch_size', 16)
        self.num_workers = get_or_default(trn_params, 'num_workers', 2)
        self.aug_type = get_or_default(cfg, 'aug', '0')
        self.csv_path = get_or_default(cfg, 'csv_path', 'input/train_folds.csv')
        self.trn_path = get_or_default(cfg, 'image_path', 'input/train_merged')
        self.mixup = get_or_default(cfg, 'mixup', False)
        self.do_cutmix = False

        self.disc_is_healty = nn.Sequential(
            nn.Linear(self.model.n_out, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 1),
            RevGrad(),
        )

        if 'crit' not in cfg:
            self.crit = nn.CrossEntropyLoss()
        elif cfg['crit'] == 'focal':
            self.crit = FocalLoss()
        elif cfg['crit'] == 'smooth':
            self.crit = SmoothCrossEntropyLoss()
        elif cfg['crit'] == 'cutmix':
            self.crit = CutMixCrossEntropyLoss(True)
            self.do_cutmix = True
        else:
            raise Exception('criterion not specified')
        print('mixup', self.mixup)

        print('using fold', self.fold)

    def forward(self, x):
        x = self.model.backbone.forward_features(x)
        x = self.model.pool(x).squeeze()
        return self.model.head(x), self.disc_is_healty(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, y_is_healthy = self.forward(x)
        y_h_gt = torch.zeros_like(y_is_healthy).squeeze()
        y_h_gt[torch.where(y == 4)[0]] = 1
        y_h_gt = y_h_gt.unsqueeze(1)
        loss = self.crit(y_hat, y)
        loss_is_healthy = F.binary_cross_entropy_with_logits(y_is_healthy, y_h_gt)
        loss = loss + 0.1 * loss_is_healthy

        self.log('trn/_loss_is_healthy', loss_is_healthy.item())
        self.log('trn/_loss', loss)
        acc = accuracy_score(y.detach().cpu().numpy(), np.argmax(y_hat.detach().cpu().numpy(), axis=1))
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
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif self.cfg['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif self.cfg['optimizer'] == 'adamw_gcc2':
            optimizer = AdamW_GCC2(self.parameters(), lr=lr)
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
        sched = {'scheduler': sched, 'name': format(self.cfg['scheduler']['type'])}
        return [optimizer], [sched]

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
    args = parser.parse_args()

    cfg = benedict.from_yaml(args.config)
    module = CassavaModule(cfg=cfg)

    early_stop = EarlyStopping(monitor='val/avg_acc', verbose=True, patience=200, mode='max')
    logger = TensorBoardLogger("lightning_logs", name=args.config)
    lrm = LearningRateMonitor()
    mdl_ckpt = ModelCheckpoint(monitor='val/avg_acc', save_top_k=3, )
    trainer = pl.Trainer(gpus=1, max_epochs=200, callbacks=[early_stop, lrm, mdl_ckpt],
                         logger=logger, gradient_clip_val=1.0, precision=16)

    trainer.fit(module)
