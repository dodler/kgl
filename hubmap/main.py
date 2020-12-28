import argparse

from benedict import benedict
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from grad_cent import Adam_GCC, AdamW_GCC2
from hubmap.aug import get_aug
from hubmap.data import HuBMAPDataset
from lovasz import lovasz_hinge
from seed import seed_everything

SEED = 2020
seed_everything(SEED)


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

        backbone = cfg['model']['backbone']
        encoder_weights = get_or_default(cfg['model'], 'weights', 'imagenet')

        if cfg['model']['type'] == 'Unet':
            self.model = smp.Unet(encoder_name=backbone, classes=1, encoder_weights=encoder_weights)
        else:
            raise Exception(cfg['model']['name'] + ' not supported')
        self.crit = symmetric_lovasz

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
        ds = HuBMAPDataset(tfms=get_aug(), train_path=self.train_path, mask_path=self.mask_path)
        dl = DataLoader(ds, batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.num_workers)

        return dl

    def val_dataloader(self):
        ds = HuBMAPDataset(train_path=self.train_path, mask_path=self.mask_path)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dl


if __name__ == '__main__':
    nfolds = 4
    fold = 0
    LABELS = '../input/hubmap-kidney-segmentation/train.csv'
    NUM_WORKERS = 4

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
