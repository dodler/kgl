import argparse

import timm
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from benedict import benedict
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from cassava.smoothed_loss import SmoothCrossEntropyLoss
from grad_cent import AdamW_GCC2
from pretrain_medical.data import PretrainDs
from seed import seed_everything

SEED = 2020
seed_everything(SEED)


def get_or_default(d, key, default_value):
    if key in d:
        return d[key]
    else:
        return default_value


class PretrainModule(pl.LightningModule):

    def __init__(self, cfg, fold=0):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(cfg['backbone'], pretrained=False, num_classes=3)
        trn_params = cfg['train_params']
        self.fold = fold
        self.batch_size = get_or_default(trn_params, 'batch_size', 16)
        self.num_workers = get_or_default(trn_params, 'num_workers', 2)
        self.crit = SmoothCrossEntropyLoss(smoothing=0.05)
        self.df = pd.read_csv('/home/lyan/Documents/kaggle/pretrain/medical_pretrain.csv')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)
        loss = self.crit(y_hat, y)

        self.log('trn/_loss', loss)
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
        lr = float(get_or_default(self.cfg['train_params'], key='lr', default_value=5e-4))
        optimizer = AdamW_GCC2(self.model.parameters(), lr=lr)
        self.opt = optimizer

        T_mult = 2
        T_0 = 10
        eta_min = 1e-8
        sched = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=-1)
        sched = {'scheduler': sched, 'name': 'CosineAnnealingWarmRestarts'}

        return [optimizer], [sched]

    def train_dataloader(self):
        train = self.df[self.df.fold != self.fold].reset_index().drop('index', axis=1).drop('fold', axis=1)
        trn_ds = PretrainDs(df=train)

        trn_dl = torch.utils.data.DataLoader(trn_ds, shuffle=True,
                                             batch_size=self.batch_size,
                                             num_workers=self.num_workers)
        return trn_dl

    def val_dataloader(self):
        valid = self.df[self.df.fold == self.fold].reset_index().drop('index', axis=1).drop('fold', axis=1)
        trn_ds = PretrainDs(df=valid)

        val_dl = torch.utils.data.DataLoader(trn_ds, shuffle=False,
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
    module = PretrainModule(cfg=cfg, fold=args.fold)

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
