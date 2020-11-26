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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from cassava.data import CassavaDs
from cassava.model import CassavaModel

train = pd.read_csv('input/train.csv')
trn, val = train_test_split(train, test_size=0.1)
trn = trn.reset_index().drop('index', axis=1)
val = val.reset_index().drop('index', axis=1)

SIZE = 600

trn_aug = alb.Compose([
    alb.Resize(SIZE, SIZE, p=1),
    alb.Transpose(p=0.5),
    alb.HorizontalFlip(p=0.5),
    alb.VerticalFlip(p=0.5),
    alb.ShiftScaleRotate(p=0.5),
    alb.Normalize(p=1),
    ToTensorV2(p=1),
])
val_aug = alb.Compose([
    alb.Resize(SIZE, SIZE, p=1),
    alb.Normalize(p=1),
    ToTensorV2(p=1),
])

trn_path = 'input/train_images'
batch_size = 16
num_workers = 2


class CassavaModule(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = CassavaModel(backbone=cfg['backbone'])
        self.crit = nn.CrossEntropyLoss(weight=torch.tensor([2, 2, 2, 1, 2]).float().cuda())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)
        loss = self.crit(y_hat, y)
        acc = accuracy_score(y.detach().cpu().numpy(), np.argmax(y_hat.detach().cpu().numpy(), axis=1))

        self.log('trn/_loss', loss)
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
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif self.cfg['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        else:
            raise Exception('optimizer {} not supported'.format(self.cfg['optimizer']))

        self.opt = optimizer

        if self.cfg['scheduler']['type'] == 'CosineAnnealingWarmRestarts':
            T_mult = self.cfg['scheduler']['T_mult']
            T_0 = self.cfg['scheduler']['T_0']
            sched = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6, last_epoch=-1)
        sched = {'scheduler': sched, 'name': 'adam+{}'.format(self.cfg['scheduler']['type'])}
        return [optimizer], [sched]

    def train_dataloader(self):
        trn_ds = CassavaDs(df=trn, aug=trn_aug, path=trn_path)

        trn_dl = torch.utils.data.DataLoader(trn_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        return trn_dl

    def val_dataloader(self):
        val_ds = CassavaDs(df=val, aug=val_aug, path=trn_path)
        val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        return val_dl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, required=False)
    args = parser.parse_args()

    cfg = benedict.from_yaml(args.config)
    module = CassavaModule(cfg=cfg)

    early_stop = EarlyStopping(monitor='val/avg_acc', verbose=True, patience=30, mode='max')
    lrm = LearningRateMonitor()
    mdl_ckpt = ModelCheckpoint(monitor='val/avg_acc', save_top_k=5, )
    trainer = pl.Trainer(gpus=1, max_epochs=200, callbacks=[early_stop, lrm, mdl_ckpt])

    trainer.fit(module)
