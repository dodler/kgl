import argparse

import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from benedict import benedict
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import Accuracy

from covid_det.aug import get_train_aug, get_valid_aug
from covid_det.data import DatasetRetriever
from covid_det.main import CovidDetModuleBase
from utils import get_or_default

import torch.nn as nn

import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        return lambda_ * grads.neg(), None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class CovidClsModule(CovidDetModuleBase):

    def __init__(self, cfg, fold=0):
        super().__init__(cfg=cfg, fold=fold)

        trn_params = cfg['train_params']
        backbone = get_or_default(cfg['model'], 'backbone', 'tf_efficientnet_b4_ns')
        self.model = timm.create_model(backbone, pretrained=True, num_classes=4, in_chans=1)

        self.grad_starvation_fix = bool(get_or_default(trn_params, 'grad_starvation_fix', False))

        self.dom_head = nn.Sequential(
            GradientReversal(),
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.ReLU(),
        )
        self.acc_metric = Accuracy()
        self.df = pd.read_csv(get_or_default(cfg['train_params'], 'csv', None))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, y_dom = batch

        b, c, h, w = x.shape
        x = x.expand(b, 3, h, w)

        features = self.model.forward_features(x)
        features = F.adaptive_avg_pool2d(features, 1).squeeze()
        y_hat = self.model.classifier(features)

        y_hat_dom = self.dom_head(features)

        loss = F.cross_entropy(input=y_hat, target=y) + F.cross_entropy(input=y_hat_dom, target=y_dom) * 0.05
        if self.grad_starvation_fix:
            loss = loss + (y_hat ** 2).mean() * 1e-2
        acc = self.acc_metric(torch.softmax(y_hat, 1), y)

        self.log('trn/_loss', loss)
        self.log('trn/_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        b, c, h, w = x.shape
        x = x.expand(b, 3, h, w)

        features = self.model.forward_features(x)
        features = F.adaptive_avg_pool2d(features, 1).squeeze()
        y_hat = self.model.classifier(features)

        loss = F.cross_entropy(input=y_hat, target=y)
        acc = self.acc_metric(torch.softmax(y_hat, 1), y)

        self.log('val/_loss', loss)
        self.log('val/_acc', acc)
        return acc

    def validation_epoch_end(self, outputs):
        avg_acc = torch.tensor(outputs).mean()

        self.log('val/_avg_acc', avg_acc)

    def train_dataloader(self):
        trn_df = self.df[(self.df.fold != self.fold) | (pd.isna(self.df.fold))].reset_index().drop('index', axis=1)
        trn_aug = get_train_aug(name=None, size=self.img_size)
        trn_ds = DatasetRetriever(df=trn_df, aug=trn_aug, cls_mode=True)
        return torch.utils.data.DataLoader(trn_ds, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        val_df = self.df[self.df.fold == self.fold].reset_index().drop('index', axis=1)
        val_aug = get_valid_aug(name=None, size=self.img_size)
        trn_ds = DatasetRetriever(df=val_df, aug=val_aug, cls_mode=True)
        return torch.utils.data.DataLoader(trn_ds, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, required=False)
    parser.add_argument('--fold', type=int, required=False, default=0)
    args = parser.parse_args()

    cfg = benedict.from_yaml(args.config)
    module = CovidClsModule(cfg=cfg, fold=args.fold)

    mode = 'max'
    tag = 'val/_avg_acc'
    early_stop = EarlyStopping(monitor=tag, verbose=True, patience=20, mode=mode)
    logger = TensorBoardLogger("lightning_logs", name=args.config)
    lrm = LearningRateMonitor()
    mdl_ckpt = ModelCheckpoint(monitor=tag, save_top_k=3, mode=mode)
    precision = get_or_default(cfg, 'precision', 32)
    grad_clip = float(get_or_default(cfg, 'grad_clip', 0))
    trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[early_stop, lrm, mdl_ckpt],
                         logger=logger, precision=precision, gradient_clip_val=grad_clip)

    trainer.fit(module)
