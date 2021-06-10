import argparse

import torch.nn.functional as F
import albumentations as alb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2
from aug import get_aug
from benedict import benedict
from data import CassavaDs
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from cassava.pazzle_mix.model_pazzle_mix import CassavaModel
from opts.grad_cent import AdamW_GCC2


def get_or_default(d, key, default_value):
    if key in d:
        return d[key]
    else:
        return default_value


SNAPMIX_ALPHA = 5.0
SNAPMIX_PCT = 1.0
device = 0


class CassavaModule(pl.LightningModule):

    def __init__(self, cfg, fold):
        super().__init__()
        self.fold = fold
        self.cfg = cfg
        self.model = CassavaModel(cfg=cfg)
        self.crit = nn.BCELoss()
        trn_params = cfg['train_params']
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

        self.model.eval()
        y_hat, _, _ = self.model(x, None)
        loss = F.cross_entropy(input=y_hat, target=y)
        loss.backward(retain_graph=True)
        unary = torch.sqrt(torch.mean(x.grad ** 2, dim=1))

        self.model.train()
        self.opt.zero_grad()
        y_hat, feats, target_reweighted = self.model(x, y, grad=unary)
        loss = self.crit(torch.softmax(y_hat), target_reweighted)

        self.manual_backward(loss=loss, optimizer=self.opt, retain_graph=True)
        self.opt.step()

        acc = accuracy_score(y.detach().cpu().numpy(), np.argmax(y_hat.detach().cpu().numpy(), axis=1))

        self.log('trn/_loss', loss)
        self.log('trn/_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _, _ = self.forward(x)
        acc = accuracy_score(y.detach().cpu().numpy(), np.argmax(y_hat.detach().cpu().numpy(), axis=1))

        self.log('val/_acc', acc, prog_bar=True)

        return acc

    def validation_epoch_end(self, outputs):
        avg_acc = np.array(outputs).mean()
        self.log('val/avg_acc', avg_acc)

    def configure_optimizers(self):
        lr = float(self.cfg['train_params']['lr'])
        optimizer = AdamW_GCC2(self.model.parameters(), lr=lr)

        self.opt = optimizer

        if self.cfg['scheduler']['type'] == 'CosineAnnealingWarmRestarts':
            T_mult = self.cfg['scheduler']['T_mult']
            T_0 = self.cfg['scheduler']['T_0']
            eta_min = float(self.cfg['scheduler']['eta_min'])
            sched = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=-1)

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
    parser.add_argument('--fold', type=int, required=False, default=0)
    args = parser.parse_args()

    cfg = benedict.from_yaml(args.config)
    module = CassavaModule(cfg=cfg, fold=args.fold)

    early_stop = EarlyStopping(monitor='val/avg_acc', verbose=True, patience=10, mode='max')
    output_name = args.config + '_fold_' + str(args.fold)
    logger = TensorBoardLogger("lightning_logs", name=output_name)
    lrm = LearningRateMonitor()
    mdl_ckpt = ModelCheckpoint(monitor='val/avg_acc', save_top_k=1, mode='max')
    precision = get_or_default(cfg['train_params'], key='precision', default_value=32)
    clip_grad = get_or_default(cfg['train_params'], key='clip_grad', default_value=0.0)
    trainer = pl.Trainer(gpus=1, max_epochs=60, callbacks=[early_stop, lrm, mdl_ckpt], logger=logger,
                         precision=precision, gradient_clip_val=clip_grad, automatic_optimization=False)

    trainer.fit(module)
