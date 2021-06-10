import argparse

import albumentations as alb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2
from benedict import benedict
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from opts.grad_cent import AdamW_GCC2
from ranzcr.aug import get_aug
from ranzcr.data import RanzcrDs
from ranzcr.model import RanzcrModel
from seed import seed_everything
from utils import get_or_default, mixup_data, mixup_criterion

SEED = 2020
seed_everything(SEED)


class RanzcrModule(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = RanzcrModel(cfg=cfg)
        trn_params = cfg['train_params']
        self.fold = get_or_default(trn_params, 'fold', 0)
        self.batch_size = get_or_default(trn_params, 'batch_size', 16)
        self.num_workers = get_or_default(trn_params, 'num_workers', 2)
        self.aug_type = get_or_default(cfg, 'aug', '0')
        self.csv_path = get_or_default(cfg, 'csv_path', 'input/train_folds.csv')
        self.trn_path = get_or_default(cfg, 'image_path', 'input/train')

        self.crit = nn.BCEWithLogitsLoss()
        print('using fold', self.fold)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        x, y_a, y_b, lam = mixup_data(x, y)
        y_hat = self.forward(x)
        loss = mixup_criterion(self.crit, y_hat, y_a.float(), y_b.float(), lam)
        self.log('trn/_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        y_hat = self.forward(x)

        return y_hat.detach().cpu().numpy(), y.detach().cpu().numpy()

    def validation_epoch_end(self, outputs):
        predictions = np.concatenate([x[0] for x in outputs])
        gt = np.concatenate([x[1] for x in outputs])

        scores = []
        for i in range(11):
            try:
                ra = roc_auc_score(y_true=gt[:, i], y_score=predictions[:, i])
            except:
                ra = 0
            scores.append(ra)

        self.log('val/_avg_roc_auc', np.array(scores).mean())

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

        trn_aug = get_aug(cfg=self.cfg)

        train = pd.read_csv(self.csv_path)
        trn_ds = RanzcrDs(df=train[train.fold != self.fold].reset_index().drop('index', axis=1).drop('fold', axis=1),
                          aug=trn_aug, path=self.trn_path)

        trn_dl = torch.utils.data.DataLoader(trn_ds, shuffle=True,
                                             batch_size=self.batch_size,
                                             num_workers=self.num_workers)
        return trn_dl

    def val_dataloader(self):
        val_aug = alb.Compose([
            alb.Resize(self.cfg['aug']['size'], self.cfg['aug']['size'], p=1),
            alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1),
        ])

        train = pd.read_csv(self.csv_path)

        val_ds = RanzcrDs(df=train[train.fold == self.fold].reset_index().drop('index', axis=1).drop('fold', axis=1),
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
    module = RanzcrModule(cfg=cfg)

    early_stop = EarlyStopping(monitor='val/_avg_roc_auc', verbose=True, patience=20, mode='max')
    logger = TensorBoardLogger("lightning_logs", name=args.config)
    lrm = LearningRateMonitor()
    mdl_ckpt = ModelCheckpoint(monitor='val/_avg_roc_auc', save_top_k=3, mode='max')
    precision = get_or_default(cfg, 'precision', 32)
    grad_clip = float(get_or_default(cfg, 'grad_clip', 0))
    trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[early_stop, lrm, mdl_ckpt],
                         logger=logger, precision=precision, gradient_clip_val=grad_clip)

    trainer.fit(module)
