import argparse

import albumentations as alb
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2
from benedict import benedict
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from opt.radam import RAdam
from opts.grad_cent import AdamW_GCC2
from seed import seed_everything
from shopee.aug import get_aug
from shopee.data import ShopeeDs
from shopee.model import ShopeeModelTimm, ShopeeModelResnext
from utils import get_or_default

SEED = 2020
seed_everything(SEED)


class ShopeeModule(pl.LightningModule):

    def __init__(self, cfg, fold=0):
        super().__init__()
        self.fold = fold
        self.cfg = cfg
        trn_params = cfg['train_params']

        self.batch_size = get_or_default(trn_params, 'batch_size', 16)
        self.num_workers = get_or_default(trn_params, 'num_workers', 2)
        self.aug_type = get_or_default(cfg, 'aug', '0')
        self.margin_start = get_or_default(cfg, 'margin_start', 10)
        self.csv_path = get_or_default(cfg, 'csv_path', 'input/train_folds.csv')
        self.trn_path = get_or_default(cfg, 'image_path', 'input/train')
        self.le = LabelEncoder()
        train = pd.read_csv(self.csv_path)
        self.le.fit(train.label_group)
        model = get_or_default(cfg, 'model', 'ShopeeModelTimm')
        self.model = model
        num_classes = len(self.le.classes_)

        if model == 'ShopeeModelTimm':
            self.model = ShopeeModelTimm(num_classes, backbone=cfg['backbone'])
        elif model == 'ShopeeModelResnext':
            self.model = ShopeeModelResnext(num_classes=num_classes)
        else:
            raise Exception('unsupported model {}'.format(model))

        self.crit = nn.CrossEntropyLoss()
        self.acc = Accuracy()
        print('using fold', self.fold)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        if self.current_epoch > self.margin_start:
            y_hat = self.model.forward(x, y)
        else:
            y_hat = self.model.forward(x)

        loss = self.crit(input=y_hat, target=y)
        if self.model == 'ShopeeModelResnext':
            th = sum(self.model.backbone.thomson_losses) / self.batch_size

            loss = loss + th
            self.log('trn/_th', th.item())
        self.log('trn/_loss', loss)
        self.log('trn/_acc', self.acc(torch.softmax(y_hat), y), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        y_hat = self.model.forward(x)
        return y_hat, y

    def validation_epoch_end(self, outputs):
        predictions = torch.cat([x[0] for x in outputs])
        target = torch.cat([x[1] for x in outputs])

        self.log('val/_avg_acc', self.acc(predictions, target))

    def configure_optimizers(self):

        lr = float(self.cfg['train_params']['lr'])

        if self.cfg['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif self.cfg['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif self.cfg['optimizer'] == 'adamw_gcc2':
            optimizer = AdamW_GCC2(self.parameters(), lr=lr)
        elif self.cfg['optimizer'] == 'radam':
            optimizer = RAdam(self.model.parameters())
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

        trn_aug = get_aug(atype=self.cfg['aug']['type'], size=self.cfg['aug']['size'])

        train = pd.read_csv(self.csv_path)
        train = train[train.fold != self.fold].reset_index().drop('index', axis=1).drop('fold', axis=1)

        images = train.image.values
        labels = self.le.transform(train.label_group.values)

        trn_ds = ShopeeDs(path=self.trn_path, images=images, labels=labels, aug=trn_aug)
        trn_dl = torch.utils.data.DataLoader(trn_ds, shuffle=True,
                                             batch_size=self.batch_size,
                                             num_workers=self.num_workers)
        return trn_dl

    def val_dataloader(self):
        val_aug = alb.Compose([
            alb.Resize(self.cfg['aug']['size'], self.cfg['aug']['size'], p=1),
            alb.Normalize(p=1.0),
            ToTensorV2(p=1),
        ])

        train = pd.read_csv(self.csv_path)
        train = train[train.fold == self.fold].reset_index().drop('index', axis=1).drop('fold', axis=1)

        images = train.image.values
        labels = self.le.transform(train.label_group.values)

        val_ds = ShopeeDs(path=self.trn_path, images=images, labels=labels, aug=val_aug)
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
    module = ShopeeModule(cfg=cfg, fold=args.fold)

    early_stop = EarlyStopping(monitor='val/_avg_acc', verbose=True, patience=20, mode='max')
    logger = TensorBoardLogger("lightning_logs", name=args.config)
    lrm = LearningRateMonitor()
    mdl_ckpt = ModelCheckpoint(monitor='val/_avg_acc', save_top_k=3, mode='max')
    precision = get_or_default(cfg, 'precision', 32)
    grad_clip = float(get_or_default(cfg, 'grad_clip', 0))
    print('using precision', precision, 'and grad clip', grad_clip)
    trainer = pl.Trainer(gpus=-1, max_epochs=100, callbacks=[early_stop, lrm, mdl_ckpt],
                         logger=logger, precision=precision, gradient_clip_val=grad_clip, sync_batchnorm=True)

    trainer.fit(module)
