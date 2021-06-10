import argparse
import os

import random
import cv2
import pandas as pd
import pytorch_lightning as pl
import torch
from benedict import benedict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

import numpy as np
from covid_det.aug import get_train_aug, get_valid_aug
from covid_det.data import DatasetRetriever
from opts.cosangulargrad import cosangulargrad
from opts.grad_cent import AdamW_GCC2
from utils import get_or_default


def collate_fn(batch):
    return tuple(zip(*batch))


IMG_SIZE = 1024


def get_train_efficientdet():
    config = get_efficientdet_config('tf_efficientdet_d5')
    config.num_classes = 3
    config.image_size = (IMG_SIZE, IMG_SIZE)
    config.norm_kwargs = dict(eps=.001, momentum=.01)
    net = EfficientDet(config, pretrained_backbone=False)
    #     checkpoint = torch.load('../input/efficientdet/efficientdet_d5-ef44aea8.pth')
    #     net.load_state_dict(checkpoint)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    return DetBenchTrain(net, config)


def plot_rec(img, rectangles):
    for rec in rectangles:
        x, y, x2, y2 = rec

        cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2, -1)
    return img


class CovidDetModuleBase(pl.LightningModule):

    def vis(self, image, target):
        img = image[0].clone()
        for t, m, s in zip(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)):
            t.mul_(s).add_(m)

        img = img.detach().cpu().numpy().transpose(1, 2, 0) * 255
        img = img.astype(np.uint8).copy()
        boxes = target[0]['boxes']
        img = plot_rec(img, boxes)
        img = torch.from_numpy(img).permute(2, 0, 1)

        self.logger.experiment.add_image('img/_in_images', img, self.global_step)

    def __init__(self, cfg, fold):
        super().__init__()
        LOCAL_RANK = int(os.environ.get("GLOBAL_RANK", 0))
        seed_everything(42 + LOCAL_RANK)
        self.cfg = cfg
        self.model = get_train_efficientdet()
        self.fold = fold
        self.batch_size = 16
        self.num_workers = 4
        self.df = pd.read_csv('train_folds_only_pos.csv')
        trn_params = cfg['train_params']
        self.img_size = get_or_default(trn_params, 'img_size', 512)

        print('using fold', self.fold)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        size = torch.tensor([(IMG_SIZE, IMG_SIZE)])
        scale = torch.tensor([1.])

        targets2 = {
            "bbox": [torch.from_numpy(target["boxes"]).cuda().float() for target in targets],
            "cls": [target["labels"].float() for target in targets],
            "img_size": torch.cat([size for target in targets]).cuda().float(),
            "img_scale": torch.cat([scale for target in targets]).cuda().float(),
        }
        images = torch.cat(images).unsqueeze(1)
        b, c, h, w = images.shape
        images = images.expand(b, 3, h, w)

        if random.random() > 0.9:
            self.vis(image=images, target=targets)

        losses_dict = self.model(images, target=targets2)

        self.log('trn/_loss', losses_dict['loss'])

        return losses_dict['loss']

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        size = torch.tensor([(IMG_SIZE, IMG_SIZE)])
        scale = torch.tensor([1.])

        targets2 = {
            "bbox": [torch.from_numpy(target["boxes"]).cuda().float() for target in targets],
            "cls": [target["labels"].float() for target in targets],
            "img_size": torch.cat([size for target in targets]).cuda().float(),
            "img_scale": torch.cat([scale for target in targets]).cuda().float(),
        }

        images = torch.cat(images).unsqueeze(1)
        b, c, h, w = images.shape
        images = images.expand(b, 3, h, w)
        losses_dict = self.model(images, target=targets2)

        self.log('trn/_loss', losses_dict['loss'])

        return losses_dict['loss']

    def configure_optimizers(self):

        lr = float(self.cfg['train_params']['lr'])

        if self.cfg['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif self.cfg['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif self.cfg['optimizer'] == 'adamw_gcc2':
            optimizer = AdamW_GCC2(self.parameters(), lr=lr)
        elif self.cfg['optimizer'] == 'cosangulargrad':
            optimizer = cosangulargrad(self.parameters(), lr=lr)
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
        trn_df = self.df[self.df.fold != self.fold].reset_index().drop('index', axis=1)
        trn_aug = get_train_aug(name=None, size=self.img_size)
        trn_ds = DatasetRetriever(df=trn_df, aug=trn_aug)
        return torch.utils.data.DataLoader(trn_ds, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        val_df = self.df[self.df.fold == self.fold].reset_index().drop('index', axis=1)
        val_aug = get_valid_aug(name=None, size=self.img_size)
        trn_ds = DatasetRetriever(df=val_df, aug=val_aug)
        return torch.utils.data.DataLoader(trn_ds, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers, collate_fn=collate_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, required=False)
    parser.add_argument('--fold', type=int, required=False, default=0)
    args = parser.parse_args()

    cfg = benedict.from_yaml(args.config)
    module = CovidDetModuleBase(cfg=cfg, fold=args.fold)

    mode = 'min'
    early_stop = EarlyStopping(monitor='val/_loss', verbose=True, patience=20, mode=mode)
    logger = TensorBoardLogger("lightning_logs", name=args.config)
    lrm = LearningRateMonitor()
    mdl_ckpt = ModelCheckpoint(monitor='val/_loss', save_top_k=3, mode=mode)
    precision = get_or_default(cfg, 'precision', 32)
    grad_clip = float(get_or_default(cfg, 'grad_clip', 0))
    trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[early_stop, lrm, mdl_ckpt],
                         logger=logger, precision=precision, gradient_clip_val=grad_clip)

    trainer.fit(module)
