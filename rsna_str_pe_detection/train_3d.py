import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

import rsna_str_pe_detection.aug_3d as a3d
import rsna_str_pe_detection.data3d as d3d
import monai
from rsna_str_pe_detection.model import RsnaStrModel

# DIR_INPUT = '/var/ssd_2t_1/kaggle_rsna_str_pe/'
DIR_INPUT = '/var/ssd_1t/kaggle_rsna_str_pe/'

device = 'cuda'
if not torch.cuda.is_available():
    device = 'cpu'


class RsnaStrPeModule(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.opt = None
        self.sched = None

        self.hparams.fold = 0
        self.hparams.warm_up_step = 2

        # self.model = RsnaStrModel()
        out_dim = 9
        self.model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=3, out_channels=out_dim).to(device)
        for p in self.model.parameters():
            p.requires_grad = True
        self.data = pd.read_csv('{}train_fold.csv'.format(DIR_INPUT))
        # self.data = pd.read_csv('{}train_debug.csv'.format(DIR_INPUT))

    def forward(self, x):
        return self.model(x)

    def get_lr(self):
        """ Returns current learning rate for schedulers """

        if self.opt is None:
            raise ValueError('No learning rate schedulers initialized')
        else:
            for pg in self.opt.param_groups:
                return pg['lr']

    def training_step(self, data, batch_nb):

        x, y = data
        y = y.float()

        x = x.to(device)
        y = y.to(device)

        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        tensorboard_logs = {
            'train/_loss': loss,
            'train/_lr': self.get_lr(),
        }

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, data, batch_nb):

        x, y = data
        y = y.float()
        x = x.to(device)
        y = y.to(device)

        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        return {'avg_val_loss': avg_loss, 'log': {'valid/_avg_loss': avg_loss}}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=5e-4)

        def lr_foo(epoch):
            if epoch < self.hparams.warm_up_step:
                # warm up lr
                lr_scale = 0.1 ** (self.hparams.warm_up_step - epoch)
            else:
                lr_scale = 0.95 ** epoch

            return lr_scale

        scheduler = LambdaLR(
            opt,
            lr_lambda=lr_foo
        )

        self.sched = scheduler
        self.opt = opt
        return [opt], [scheduler]

    def train_dataloader(self):
        aug = a3d.get_rsna_train_aug('v0')

        fold_data = self.data[self.data.fold != self.hparams.fold]
        fold_data = fold_data.reset_index().drop('index', axis=1)
        train_dataset = d3d.RSNADataset3D(
            df=fold_data,
            transform=aug,
            path='{}/train-jpegs/'.format(DIR_INPUT),
        )

        sampler = RandomSampler(
            train_dataset,
            num_samples=300000,
            replacement=True,
        )

        return DataLoader(train_dataset, sampler=sampler, batch_size=4, num_workers=6)

    def val_dataloader(self):
        aug = a3d.get_rsna_valid_aug('v0')

        fold_data = self.data[self.data.fold == self.hparams.fold]
        fold_data = fold_data.reset_index().drop('index', axis=1)

        dataset = d3d.RSNADataset3D(
            df=fold_data,
            transform=aug,
            path='{}/train-jpegs/'.format(DIR_INPUT),
        )

        return DataLoader(dataset, shuffle=False, batch_size=4, num_workers=6)


if __name__ == '__main__':
    module = RsnaStrPeModule()

    checkpoint_callback = ModelCheckpoint(
        filepath='/var/data/rsna_checkpoints',
        save_top_k=5,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        prefix=''
    )

    early_stop = EarlyStopping(monitor='avg_val_loss', verbose=True, patience=10, mode='min')
    trainer = pl.Trainer(gpus=1, max_epochs=50,
                         default_root_dir='/var/data/rsna_checkpoints/',
                         early_stop_callback=early_stop,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(module)
