import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, RandomSampler

import pandas as pd
import torch.nn.functional as F
from google_landmarks2.gld_data import GldData
from google_landmarks2.model import EffB0Arc
from google_landmarks2.preproc import train_aug, valid_aug
from torch.utils.data.sampler import Sampler

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

device = 0


class EffB0ArcLightning(pl.LightningModule):

    def __init__(self):
        super(EffB0ArcLightning, self).__init__()
        self.model = EffB0Arc(device=device, n_class=81313)
        self.model.to(device)

    def forward(self, x, y):
        pred, vec = self.model(x, y)
        return pred

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.to(device)
        y = y.to(device)
        logits = self.forward(x, y)

        with torch.no_grad():
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == y).sum().item() / x.shape[0]

        loss = F.cross_entropy(logits, y)

        logs = {'train_loss': loss, 'train_acc': acc}

        return {'loss': loss, 'log': logs, 'acc': acc}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.to(device)
        y = y.to(device)
        logits = self.forward(x, y)

        with torch.no_grad():
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == y).sum().item() / x.shape[0]

        loss = F.cross_entropy(logits, y)

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = np.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        df = pd.read_csv('/home/lyan/Documents/kaggle/kaggle_landmarks/train_group_folds.csv')
        le = LabelEncoder()
        le.fit_transform(df.landmark_id)
        fold = 0
        train = df[df.fold != fold]
        valid = df[df.fold == fold]

        self.train_data = GldData(df=train, aug=train_aug, label_encoder=le)
        self.valid_data = GldData(df=valid, aug=valid_aug, label_encoder=le)

    def train_dataloader(self):
        iter_size = 32 * 4000
        sampler = torch.utils.data.sampler.RandomSampler(range(iter_size))
        bs = torch.utils.data.sampler.BatchSampler(sampler, batch_size=32, drop_last=False)

        return torch.utils.data.DataLoader(self.train_data,
                                           num_workers=6,
                                           batch_sampler=bs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_data, num_workers=6, batch_size=32, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    model = EffB0ArcLightning()
    trainer = pl.Trainer()

    trainer.fit(model)
