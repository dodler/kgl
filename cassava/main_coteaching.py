import argparse
from typing import Optional, Callable

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
from sklearn.metrics import accuracy_score
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from cassava.aug import get_aug
from cassava.data import CassavaDs
from cassava.loss_coteaching import loss_coteaching, loss_coteaching_plus
from cassava.model import CassavaModel
from grad_cent import AdamW_GCC2
from seed import seed_everything

SEED = 2020
seed_everything(SEED)


def get_or_default(d, key, default_value):
    if key in d:
        return d[key]
    else:
        return default_value


# define drop rate schedule
def gen_forget_rate(fr_type='type_1'):
    forget_rate = 0.2
    num_gradual = 10
    if fr_type == 'type_1':
        rate_schedule = np.ones(200) * forget_rate
        rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)

    # if fr_type=='type_2':
    #    rate_schedule = np.ones(args.n_epoch)*forget_rate
    #    rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
    #    rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2*forget_rate, args.n_epoch-args.num_gradual)

    return rate_schedule


class CassavaModule(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = CassavaModel(cfg=cfg)
        self.model2 = CassavaModel(cfg=cfg)

        trn_params = cfg['train_params']
        self.fold = get_or_default(trn_params, 'fold', 0)
        self.batch_size = get_or_default(trn_params, 'batch_size', 16)
        self.num_workers = get_or_default(trn_params, 'num_workers', 2)
        self.aug_type = get_or_default(cfg, 'aug', '0')
        self.csv_path = get_or_default(cfg, 'csv_path', 'input/train_folds.csv')
        self.trn_path = get_or_default(cfg, 'image_path', 'input/train_merged')
        self.mixup = get_or_default(cfg, 'mixup', False)
        self.do_cutmix = False

        self.crit = nn.CrossEntropyLoss()
        self.init_epoch = 10
        print('mixup', self.mixup)
        self.rate_schedule = gen_forget_rate()

        print('using fold', self.fold)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, indices = batch
        ind = indices.cpu().numpy().transpose()
        y_hat = self.model(x)
        y_hat2 = self.model2(x)
        noise_or_not = [0] * len(self.trn_ds)
        noise_or_not = np.array(noise_or_not)

        if self.current_epoch < self.init_epoch:
            loss_1, loss_2, _, _ = loss_coteaching(y_hat, y_hat2, y, self.rate_schedule[self.current_epoch], ind,
                                                   noise_or_not)
        else:
            loss_1, loss_2, _, _ = loss_coteaching_plus(y_hat, y_hat2, y,
                                                        self.rate_schedule[self.current_epoch], ind,
                                                        noise_or_not, self.current_epoch * batch_idx)

        self.log('trn/_loss_1', loss_1)
        self.log('trn/_loss_2', loss_2)
        acc = accuracy_score(y.detach().cpu().numpy(), np.argmax(y_hat.detach().cpu().numpy(), axis=1))
        self.log('trn/_acc', acc, prog_bar=True)
        return loss_1 + loss_2

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
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

        optimizer = AdamW_GCC2(list(self.model.parameters()) + list(self.model2.parameters()), lr=lr)

        T_mult = self.cfg['scheduler']['T_mult']
        T_0 = self.cfg['scheduler']['T_0']
        eta_min = float(self.cfg['scheduler']['eta_min'])
        sched = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=-1)
        sched = {'scheduler': sched, 'name': format(self.cfg['scheduler']['type'])}

        return [optimizer], [sched]

    def train_dataloader(self):

        trn_aug = get_aug(atype=self.aug_type, size=self.cfg['img_size'])

        train = pd.read_csv(self.csv_path)
        trn_ds = CassavaDs(df=train[train.fold != self.fold].reset_index().drop('index', axis=1).drop('fold', axis=1),
                           aug=trn_aug, path=self.trn_path, return_index=True)
        self.trn_ds = trn_ds

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
    precision = get_or_default(cfg, 'precision', 32)
    grad_clip = float(get_or_default(cfg, 'grad_clip', 0))
    trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[early_stop, lrm, mdl_ckpt],
                         logger=logger, precision=precision, gradient_clip_val=grad_clip)

    trainer.fit(module)
