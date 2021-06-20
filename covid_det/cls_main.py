import argparse

import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import numpy as np
import torch.nn.functional as F
from benedict import benedict
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import Accuracy

from covid_det.aug import get_train_aug, get_valid_aug
from covid_det.data import DatasetRetriever
from covid_det.main import CovidDetModuleBase
from covid_det.seg_head import get_seg_head
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


def mixup_data(x, y, dom=None, mask=None, alpha=1.0, use_cuda=True):
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
    if dom is not None:
        dom_a, dom_b = dom, dom[index]
    else:
        dom_a, dom_b = None, None
    if mask is not None:
        mask_a, mask_b = mask, mask[index]
    else:
        mask_a, mask_b = None, None

    return mixed_x, y_a, y_b, lam, dom_a, dom_b, mask_a, mask_b


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


feat_map_size = {
    'tf_efficientnet_b4_ns': 1792,
    'tf_efficientnet_b5_ns': 2048,
    'tf_efficientnet_b6_ns': 2304,
}


class CovidClsModule(CovidDetModuleBase):

    def __init__(self, cfg, fold=0):
        super().__init__(cfg=cfg, fold=fold)

        trn_params = cfg['train_params']
        backbone = get_or_default(cfg['model'], 'backbone', 'tf_efficientnet_b4_ns')
        self.model = timm.create_model(backbone, pretrained=True, num_classes=4, in_chans=1)

        self.seg_supervision = bool(get_or_default(trn_params, 'seg_supervision', False))
        self.seg_supervision_weight = float(get_or_default(trn_params, 'seg_supervision_weight', 0.1))
        self.grad_starvation_fix = bool(get_or_default(trn_params, 'grad_starvation_fix', False))
        self.grad_rev = bool(get_or_default(trn_params, 'grad_rev', True))

        if 'mixup' in trn_params:
            self.mixup = True
            self.mixup_alpha = float(get_or_default(trn_params, 'mixup_alpha', 1.0))
        else:
            self.mixup = False
            self.mixup_alpha = 0.0

        self.seg_head = get_seg_head(in_channels=feat_map_size[backbone], out_channels=1)
        self.seg_crit = torch.nn.BCEWithLogitsLoss()

        self.dom_head = nn.Sequential(
            GradientReversal(),
            nn.Linear(feat_map_size[backbone], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.ReLU(),
        )
        self.acc_metric = Accuracy()
        self.df = pd.read_csv(get_or_default(cfg['train_params'], 'csv', None))
        self.crit = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, y_dom, mask = batch

        if self.mixup:
            x, y_a, y_b, lam, dom_a, dom_b, mask_a, mask_b = mixup_data(x, y, dom=y_dom, mask=mask)

        features = self.model.forward_features(x)

        if self.seg_supervision:
            y_seg_hat = self.seg_head(features)
            if self.mixup:
                seg_loss = mixup_criterion(self.seg_crit, y_seg_hat, mask_a, mask_b, lam)
            else:
                seg_loss = self.seg_crit(input=y_seg_hat, target=mask)

        features = F.adaptive_avg_pool2d(features, 1).squeeze()
        y_hat = self.model.classifier(features)

        y_hat_dom = self.dom_head(features)

        if self.mixup:
            base_loss = mixup_criterion(self.crit, y_hat, y_a, y_b, lam)
        else:
            base_loss = F.cross_entropy(input=y_hat, target=y)

        loss = base_loss
        if self.grad_rev:
            if self.mixup:
                loss_grad_rev = mixup_criterion(self.crit, y_hat_dom, dom_a, dom_b, lam)
            else:
                loss_grad_rev = self.crit(input=y_hat_dom, target=y_dom)

            loss = loss + loss_grad_rev*0.05
        if self.grad_starvation_fix:
            loss = loss + (y_hat ** 2).mean() * 1e-2
        if self.seg_supervision:
            loss = loss + seg_loss * self.seg_supervision_weight
        acc = self.acc_metric(torch.softmax(y_hat, 1), y)

        self.log('trn/_loss', loss)
        self.log('trn/_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch

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
