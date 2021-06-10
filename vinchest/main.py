import sys

import argparse

from benedict import benedict
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append('/home/jovyan/kaggle/vinxray_chest/timm-efficientdet-pytorch')

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch._six
import torch.utils.data

from vinchest.aug import get_train_transforms, get_valid_transforms
from vinchest.data import DatasetRetriever
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


def collate_fn(batch):
    return tuple(zip(*batch))


def get_net():
    config = get_efficientdet_config('tf_efficientdet_d0')
    net = EfficientDet(config, pretrained_backbone=True)
    config.num_classes = 14
    config.image_size = 512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)


class VinchestModule(pl.LightningModule):

    def __init__(self):
        super(VinchestModule, self).__init__()
        dataframe = pd.read_csv('train.csv')
        dataframe = dataframe[dataframe['class_id'] != 14].reset_index(drop=True)

        np.random.seed(0)
        image_names = np.random.permutation(dataframe.image_id.unique())
        valid_images_len = int(len(image_names) * 0.2)
        images_valid = image_names[:valid_images_len]
        images_train = image_names[valid_images_len:]
        images_valid = dataframe[dataframe.image_id.isin(images_valid)].image_id.unique()
        images_train = dataframe[~dataframe.image_id.isin(images_valid)].image_id.unique()

        self.train_dataset = DatasetRetriever(
            image_ids=images_train,
            marking=dataframe,
            transforms=get_train_transforms(),
            test=False,
        )
        self.validation_dataset = DatasetRetriever(
            image_ids=images_valid,
            marking=dataframe,
            transforms=get_valid_transforms(),
            test=True,
        )
        self.model = get_net()
        self.cfg = TrainGlobalConfig()

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=TrainGlobalConfig.batch_size,
                                                   num_workers=TrainGlobalConfig.num_workers,
                                                   shuffle=True, collate_fn=collate_fn)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(self.validation_dataset,
                                                   batch_size=TrainGlobalConfig.batch_size,
                                                   num_workers=TrainGlobalConfig.num_workers,
                                                   shuffle=False, collate_fn=collate_fn)

        return valid_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                     T_0=20,
                                                                     T_mult=2,
                                                                     eta_min=1e-8,
                                                                     last_epoch=-1)
        return [optimizer], [sched]

    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch

        images = torch.stack(images)
        boxes = [target['boxes'].float() for target in targets]
        labels = [target['labels'].float() for target in targets]
        loss, _, _ = self.model(images, boxes, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch

        images = torch.stack(images)
        boxes = [target['boxes'].float() for target in targets]
        labels = [target['labels'].float() for target in targets]
        loss, _, _ = self.model(images, boxes, labels)

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('val/avg_loss', avg_loss)


class TrainGlobalConfig:
    num_workers = 2
    batch_size = 4
    n_epochs = 2
    lr = 0.0002
    folder = 'effdet5-models'
    verbose = True
    verbose_step = 1
    step_scheduler = False
    validation_scheduler = True
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08

    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, required=False)
    parser.add_argument('--fold', type=int, required=False, default=0)
    args = parser.parse_args()

    cfg = benedict.from_yaml(args.config)
    module = VinchestModule()
    #
    output_name = args.config + '_fold_' + str(args.fold)
    logger = TensorBoardLogger("lightning_logs", name=output_name)

    early_stop = EarlyStopping(monitor='val/avg_loss', verbose=True, patience=10, mode='min')
    lrm = LearningRateMonitor()
    mdl_ckpt = ModelCheckpoint(monitor='val/avg_loss', save_top_k=3, mode='min')
    precision = 32
    grad_clip = 0.5
    epochs = 80
    trainer = pl.Trainer(gpus=1,
                         max_epochs=epochs,
                         precision=precision,
                         callbacks=[early_stop, lrm, mdl_ckpt],
                         gradient_clip_val=grad_clip)

    trainer.fit(module)

