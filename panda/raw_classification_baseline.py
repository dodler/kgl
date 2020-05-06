import argparse

import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader

from panda.panda_augs.v0 import valid_aug, train_aug
from panda.panda_dataset import PandaImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--batch-size', type=int, default=128)
args = parser.parse_args()

df = pd.read_csv('/home/lyan/Documents/kaggle/panda/train_folds.csv')
train_df = df[df.fold == args.fold]
valid_df = df[df.fold != args.fold]


class PandaModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.bn = nn.BatchNorm1d(num_features=1792)
        self.drop = nn.Dropout(p=0.2, inplace=True)
        self.lin = nn.Linear(in_features=1792, out_features=6)

        self.valid_acc_list = []
        self.train_acc_list = []

    def forward(self, x):
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.bn(x)
        x = self.drop(x)
        x = self.lin(x)
        return x.squeeze()

    @staticmethod
    def __calc_accuracy__(y, y_hat):
        _, predicted = torch.max(y_hat.data, 1)
        acc = (predicted == y).sum().item() / args.batch_size
        return acc

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x.float()
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        train_step_acc = PandaModule.__calc_accuracy__(y, y_hat)
        self.train_acc_list.append(train_step_acc)
        tb_logs = {'train_step/_acc': train_step_acc,
                   'train_step/_loss': loss.item()}

        return {'loss': loss, 'log': tb_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x = x.float()
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        val_acc_step = PandaModule.__calc_accuracy__(y, y_hat)
        self.valid_acc_list.append(val_acc_step)
        tb_logs = {'valid/_acc': val_acc_step,
                   'valid/_loss': loss.item()}

        return {'val_loss': loss, 'log': tb_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = np.array(self.valid_acc_list).mean()
        print('average valid accuracy', avg_acc, 'average loss', avg_loss.item())
        self.valid_acc_list = []
        tb_logs = {'valid_epoch/_loss': avg_loss.item(), 'valid_epoch/_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': tb_logs}

    def on_epoch_end(self):
        print('avg train accuracy', np.array(self.train_acc_list).mean())
        self.train_acc_list = []

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, min_lr=1e-5, patience=10)
        return [opt], [scheduler]

    def train_dataloader(self):
        train_ds = PandaImageDataset(df=train_df, aug=train_aug, )
        train_loader = DataLoader(train_ds, shuffle=True, num_workers=12, batch_size=args.batch_size)
        return train_loader

    def val_dataloader(self):
        valid_ds = PandaImageDataset(valid_df, aug=valid_aug)
        valid_loader = DataLoader(valid_ds, shuffle=False, num_workers=12, batch_size=args.batch_size)
        return valid_loader


panda_module = PandaModule()

trainer = pl.Trainer(gpus=1, distributed_backend='dp', max_epochs=args.epochs,
                     auto_lr_finder=True)
trainer.fit(panda_module)
