import argparse
from pathlib import Path

import albumentations as alb
import pandas as pd
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import zarr
from benedict import benedict
from deepflash2.all import *
from fastai.vision.all import *
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader

from hubmap.losses import lovasz_hinge
from opts.grad_cent import AdamW_GCC2, Adam_GCC
from seed import seed_everything
import segmentation_models_pytorch as smp
from utils import get_or_default

SEED = 2020
seed_everything(SEED)


def weighted_bce(logit_pixel, gt):
    logit = logit_pixel.view(-1)
    truth = gt.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    if 0:
        loss = loss.mean()
    if 1:
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.25 * pos * loss / pos_weight + 0.75 * neg * loss / neg_weight).sum()

    return loss


def lovasz_and_dice(pred, gt):
    return lovasz_hinge(pred, gt)


@patch
def read_img(self: BaseDataset, file, *args, **kwargs):
    return zarr.open(str(file), mode='r')


@patch
def _name_fn(self: BaseDataset, g):
    "Name of preprocessed and compressed data."
    return f'{g}'


@patch
def apply(self: DeformationField, data, offset=(0, 0), pad=(0, 0), order=1):
    "Apply deformation field to image using interpolation"
    outshape = tuple(int(s - p) for (s, p) in zip(self.shape, pad))
    coords = [np.squeeze(d).astype('float32').reshape(*outshape) for d in self.get(offset, pad)]
    # Get slices to avoid loading all data (.zarr files)
    sl = []
    for i in range(len(coords)):
        cmin, cmax = int(coords[i].min()), int(coords[i].max())
        dmax = data.shape[i]
        if cmin < 0:
            cmax = max(-cmin, cmax)
            cmin = 0
        elif cmax > dmax:
            cmin = min(cmin, 2 * dmax - cmax)
            cmax = dmax
            coords[i] -= cmin
        else:
            coords[i] -= cmin
        sl.append(slice(cmin, cmax))
    if len(data.shape) == len(self.shape) + 1:
        tile = np.empty((*outshape, data.shape[-1]))
        for c in range(data.shape[-1]):
            # Adding divide
            tile[..., c] = cv2.remap(data[sl[0], sl[1], c] / 255, coords[1], coords[0], interpolation=order,
                                     borderMode=cv2.BORDER_REFLECT)
    else:
        tile = cv2.remap(data[sl[0], sl[1]], coords[1], coords[0], interpolation=order, borderMode=cv2.BORDER_REFLECT)
    return tile


class CONFIG():
    # data paths

    # deepflash2 dataset
    scale = 1.5  # data is already downscaled to 2, so absulute downscale is 3
    tile_shape = (512, 512)
    padding = (0, 0)  # Border overlap for prediction
    n_jobs = 1
    sample_mult = 100  # Sample 100 tiles from each image, per epoch
    val_length = 500  # Randomly sample 500 validation tiles

    # deepflash2 augmentation options
    zoom_sigma = 0.1
    flip = True
    max_rotation = 360
    deformation_grid_size = (150, 150)
    deformation_magnitude = (10, 10)


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def symmetric_lovasz(outputs, targets):
    return 0.5 * (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))


def get_or_default(d, key, default_value):
    if key in d:
        return d[key]
    else:
        return default_value


class HubmapModule(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        trn_params = cfg['train_params']
        self.fold = get_or_default(trn_params, 'fold', 0)
        self.batch_size = get_or_default(trn_params, 'batch_size', 16)
        self.num_workers = get_or_default(trn_params, 'num_workers', 2)
        self.train_path = get_or_default(trn_params, 'train_path', 'input/crops256/train/')
        self.mask_path = get_or_default(trn_params, 'masks_path', 'input/crops256/masks/')

        data_path = Path(get_or_default(trn_params, 'data_path', '../input/hubmap-kidney-segmentation'))
        data_path_zarr = Path(get_or_default(trn_params, 'data_path_zarr', '../input/hubmap-zarr/train_scale2'))
        mask_preproc_dir = get_or_default(trn_params, 'mask_preproc_dir',
                                          '/kaggle/input/hubmap-labels-pdf-0-5-0-25-0-01/masks_scale2')

        backbone = cfg['model']['backbone']
        encoder_weights = get_or_default(cfg['model'], 'weights', 'imagenet')

        if cfg['model']['type'] == 'Unet':
            self.model = smp.Unet(encoder_name=backbone, classes=1, encoder_weights=encoder_weights)
        else:
            raise Exception(cfg['model']['name'] + ' not supported')
        self.crit = symmetric_lovasz

        df_train = pd.read_csv(data_path / 'train.csv')
        df_info = pd.read_csv(data_path / 'HuBMAP-20-dataset_information.csv')

        files = [x for x in data_path_zarr.iterdir() if x.is_dir() if not x.name.startswith('.')]
        label_fn = lambda o: o

        cfg = CONFIG()

        aug = alb.Compose([
            alb.OneOf([
                alb.HueSaturationValue(10, 15, 10),
                alb.CLAHE(clip_limit=2),
                alb.RandomBrightnessContrast(),
            ], p=0.3),
            alb.Normalize(p=1, std=[0.15167958, 0.23584107, 0.13146145],
                          mean=[0.65459856, 0.48386562, 0.69428385])])

        ds_kwargs = {
            'tile_shape': cfg.tile_shape,
            'padding': cfg.padding,
            'scale': cfg.scale,
            'n_jobs': cfg.n_jobs,
            'preproc_dir': mask_preproc_dir,
            'val_length': cfg.val_length,
            'sample_mult': cfg.sample_mult,
            'loss_weights': False,
            'zoom_sigma': cfg.zoom_sigma,
            'flip': cfg.flip,
            'max_rotation': cfg.max_rotation,
            'deformation_grid_size': cfg.deformation_grid_size,
            'deformation_magnitude': cfg.deformation_magnitude,
            'albumentations_tfms': aug
        }

        self.train_ds = RandomTileDataset(files, label_fn=label_fn, **ds_kwargs)
        self.valid_ds = TileDataset(files, label_fn=label_fn, **ds_kwargs, is_zarr=True)

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)
        dice = dice_coeff(pred=F.sigmoid(y_hat), target=y)
        loss = self.crit(y_hat, y)

        self.log('trn/_loss', loss)
        self.log('trn/_dice', dice, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        dice = dice_coeff(pred=F.sigmoid(y_hat), target=y)
        loss = self.crit(y_hat, y)

        self.log('val/_loss', loss)
        self.log('val/_dice', dice)

        return loss, dice

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x[0] for x in outputs]).mean()
        avg_dice = torch.stack([x[1] for x in outputs]).mean()
        self.log('val/avg_loss', avg_loss)
        self.log('val/avg_dice', avg_dice, prog_bar=True)

    def configure_optimizers(self):
        opt_cfg = self.cfg['optimizer']
        lr = float(self.cfg['optimizer']['lr'])
        if opt_cfg['name'] == 'AdamW':
            optimizer = AdamW(self.model.parameters(), lr=lr, )
        elif opt_cfg['name'] == 'Adam_GCC':
            optimizer = Adam_GCC(self.model.parameters(), lr=lr)
        elif opt_cfg['name'] == 'AdamW_GCC2':
            optimizer = AdamW_GCC2(self.model.parameters(), lr=lr)

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

        return optimizer

    def train_dataloader(self):
        dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, required=False)
    args = parser.parse_args()

    cfg = benedict.from_yaml(args.config)
    module = HubmapModule(cfg)

    early_stop = EarlyStopping(monitor='val/avg_dice', verbose=True, patience=50, mode='max')
    logger = TensorBoardLogger("lightning_logs", name=args.config)
    lrm = LearningRateMonitor()
    mdl_ckpt = ModelCheckpoint(monitor='val/avg_dice', save_top_k=5, )
    precision = get_or_default(cfg, 'precision', 32)
    clip_grad = get_or_default(cfg, 'cril_grad', 0.0)
    trainer = pl.Trainer(gpus=1, max_epochs=200, callbacks=[early_stop, lrm, mdl_ckpt], logger=logger,
                         precision=precision, gradient_clip_val=clip_grad)

    trainer.fit(module)
