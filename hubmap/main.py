import albumentations as alb
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from grad_cent import Adam_GCC
from hubmap.data import SegHubmapDs

SIZE = 384

aug = alb.Compose([
    alb.Resize(SIZE, SIZE, p=1),
    alb.Normalize(p=1),
    ToTensorV2(p=1),
])

train_ids = np.array(
    ['2f6ecfcdf', 'aaa6a05cc', 'cb2d976f4', '0486052bb', 'e79de561c', '095bf7a1f', '54f2eec69', '1e2425f28'])
# these are folds actually

batch_size = 16
num_workers = 2


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class HubmapModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnext50_32x4d", classes=1)
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)
        dice = dice_coeff(pred=F.sigmoid(y_hat), target=y)
        loss = self.crit(y_hat, y)

        self.log('trn/_loss', loss)
        self.log('trn/_dice', dice)

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
        self.log('val/avg_dice', avg_dice)

    def configure_optimizers(self):
        optimizer = Adam_GCC(self.parameters(), lr=1e-4)
        return optimizer

    def train_dataloader(self):
        val_id = train_ids[0]

        trn_ds = SegHubmapDs(aug=aug, path='input/crops/images/', train_ids=train_ids[train_ids != val_id].tolist())
        print(len(trn_ds))

        trn_dl = torch.utils.data.DataLoader(trn_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        return trn_dl

    def val_dataloader(self):
        val_ds = SegHubmapDs(aug=aug, path='input/crops/images/', train_ids=[train_ids[0]])
        val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        return val_dl


module = HubmapModule()
early_stop = EarlyStopping(monitor='val/avg_dice', verbose=True, patience=50, mode='max')
lrm = LearningRateMonitor()
mdl_ckpt = ModelCheckpoint(monitor='val/avg_dice', save_top_k=5, )
trainer = pl.Trainer(gpus=1, max_epochs=200, callbacks=[early_stop, lrm, mdl_ckpt])

trainer.fit(module)
