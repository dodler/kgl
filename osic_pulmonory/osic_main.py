import argparse
import pickle

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingWarmRestarts, ExponentialLR
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from osic_pulmonory.osic_aug import get_osic_train_aug, get_osic_valid_aug
from osic_pulmonory.osic_data import OSICData
from osic_pulmonory.osic_models import get_osic_model

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--warm_up_step', type=int, default=2000)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--model', type=str, default='b0')
parser.add_argument('--early_stop_patience', type=int, default=10, )
parser.add_argument('--sched', type=str, default='exp', choices=['cyclic', 'cosine_annealing_warm_restarts', 'exp'])
parser.add_argument('--opt', type=str, default='sgd')
parser.add_argument('--aug_name', type=str, default='v0', required=False)
args = parser.parse_args()


def score(outputs, target, device=0):
    confidence = outputs[:, 2] - outputs[:, 0]  # output -> 0-> 20% percentile, 50% percentile, 80% percentile
    clip = torch.clamp(confidence, min=70)
    target = torch.reshape(target, outputs[:, 1].shape)
    delta = torch.abs(outputs[:, 1] - target)
    delta = torch.clamp(delta, max=1000)
    sqrt_2 = torch.sqrt(torch.tensor([2.])).to(device)
    metrics = (delta * sqrt_2 / clip) + torch.log(clip * sqrt_2)
    return torch.mean(metrics)


def qloss(outputs, target, device=0):
    qs = [0.2, 0.5, 0.8]
    qs = torch.tensor(qs, dtype=torch.float).to(device)
    e = outputs - target
    e.to(device)
    v = torch.max(qs * e, (qs - 1) * e)
    return torch.mean(v)


def loss_fn(outputs, target, l):
    return l * qloss(outputs, target) + (1 - l) * score(outputs, target)


class OSICModule(pl.LightningModule):

    def __init__(self, args=None):
        super().__init__()

        self.opt = None
        self.sched = None

        if args is not None:
            for k in vars(args):
                self.hparams[k] = getattr(args, k)

        if args is not None:
            fold = args.fold
        else:
            fold = 0

        patients_df = pd.read_csv('/home/lyan/Documents/kaggle/patients.csv')
        train_patients = patients_df[patients_df.fold != fold]
        valid_patients = patients_df[patients_df.fold == fold]

        with open('tab_data.pkl', 'rb') as f:
            TAB = pickle.load(f)

        with open('A.pkl', 'rb') as f:
            A = pickle.load(f)

        self.train_aug = get_osic_train_aug(args.aug_name)
        self.valid_aug = get_osic_valid_aug(args.aug_name)

        self.train_data = OSICData(keys=train_patients, a=A, tab=TAB, aug=self.train_aug)
        self.valid_data = OSICData(keys=valid_patients, a=A, tab=TAB, aug=self.valid_aug)

        if args is not None:
            self.batch_size = args.batch_size
        else:
            self.batch_size = 1

        self.model = get_osic_model(model=args.model)

        self.val_y_hat = []
        self.val_y = []

    def forward(self, x):
        return self.model(x)

    def get_lr(self):
        ''' Returns current learning rate for schedulers '''
        if self.opt is None:
            raise ValueError('No learning rate schedulers initialized')
        else:
            for pg in self.opt.param_groups:
                return pg['lr']

    def training_step(self, batch, batch_nb):
        (x, meta), y = batch
        y = y.float()
        y = y.float().reshape(x.shape[0], 1)
        meta = meta.float().squeeze()

        y_hat = self.forward((x, meta)).unsqueeze(1)

        loss = F.mse_loss(y_hat, y, reduction='mean') + 1e-4

        tensorboard_logs = {
            'train/_loss': loss,
            'train/_lr': self.get_lr(),
        }

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        (x, meta), y = batch

        x = x.float()
        y = y.float()
        meta = meta.float().squeeze()

        y_hat = self.forward((x, meta)).unsqueeze(1)

        loss = F.mse_loss(y_hat, y) + 1e-4

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val/_loss': avg_loss}

        self.sched.step(self.current_epoch)

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):

        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        if self.hparams['opt'] == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        elif self.hparams['opt'] == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)

        if self.hparams['sched'] == 'cyclic':
            scheduler = CyclicLR(optimizer=opt, base_lr=self.hparams.lr / 500, max_lr=self.hparams.lr / 10)
        elif self.hparams['sched'] == 'cosine_annealing_warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer=opt,
                                                    T_0=2000,
                                                    eta_min=self.hparams.lr / 1000.0,
                                                    T_mult=1,
                                                    )
        elif self.hparams['sched'] == 'exp':
            scheduler_steplr = ExponentialLR(opt, gamma=0.95)
            scheduler = GradualWarmupScheduler(opt, multiplier=1, total_epoch=5,
                                               after_scheduler=scheduler_steplr)
        else:
            raise Exception('unknown scheduler {}'.format(self.hparams['sched']))

        self.sched = scheduler
        self.opt = opt
        return opt

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data, shuffle=True, num_workers=6, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_data, shuffle=False, num_workers=6, batch_size=self.batch_size)
        return valid_loader


if __name__ == '__main__':
    args = parser.parse_args()

    module = OSICModule(args=args)

    early_stop = EarlyStopping(monitor='avg_val_loss', verbose=True, patience=args.early_stop_patience, mode='min')
    trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, early_stop_callback=early_stop)
    trainer.fit(module)
