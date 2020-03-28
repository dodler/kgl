import argparse
import os
import os.path as osp
import pickle
import random
from collections import Counter

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import pretrainedmodels as pm
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, RandomSampler

from deep_fake.augs.v0 import augs
from deep_fake.deep_fake_data import ImageListDs
from deep_fake.models import get_model
import numpy as np

parser = argparse.ArgumentParser(description='Understanding cloud training')

parser.add_argument('--model',
                    default=None,
                    required=True,
                    type=str)

parser.add_argument('--aug', default='default', required=False, type=str)
parser.add_argument('--batch-size', default='64', required=False, type=int)

args = parser.parse_args()

print('using model', args.model)
print('using augmentation', args.aug)

train_aug, valid_aug = augs[args.aug]

BATCH_SIZE = args.batch_size
FAKE_LABEL = 1
TRUE_LABEL = 0

true_img_dir = '/var/ssd_1t/kaggle_deepfake/us_images_crops/'
false_img_dir = '/var/ssd_1t/nvidia_df_57g/deep_fake_nvidia_512/'

if not osp.exists('data.pkl'):
    print('collecting train data')
    fake_images = os.listdir(false_img_dir)
    fake_images = [osp.join(false_img_dir, k) for k in fake_images]
    fake_labels = [FAKE_LABEL] * len(fake_images)

    images = os.listdir(true_img_dir)
    images = [osp.join(true_img_dir, k) for k in images]
    labels = [TRUE_LABEL] * len(images)

    ppath = '/var/ssd_1t/kaggle_deepfake/train_cropped/'
    monty_images = []
    monty_labels = []

    valid_images = []
    valid_labels = []

    data = pd.read_csv('/home/lyan/Documents/kaggle/deep_fake/images_meta.csv')
    for i in range(data.shape[0]):
        label = data.iloc[i, 1]
        image_name = data.iloc[i, 5]
        prefix = data.iloc[i, 4]

        if len(valid_labels) < 20000 and random.random() > 0.95:
            valid_images.append(osp.join(ppath, prefix, image_name))
            if label == 'FAKE':
                valid_labels.append(FAKE_LABEL)
            else:
                valid_labels.append(TRUE_LABEL)

        monty_images.append(osp.join(ppath, prefix, image_name))
        if label == 'FAKE':
            monty_labels.append(FAKE_LABEL)
        else:
            monty_labels.append(TRUE_LABEL)

    p = '/var/ssd_1t/kaggle_deepfake/ru_crops/'
    ru_images = os.listdir(p)
    ru_images = [osp.join(p, k) for k in ru_images]
    ru_labels = [TRUE_LABEL] * len(ru_images)

    all_images = list(images)
    all_labels = list(labels)

    all_images.extend(fake_images)
    all_labels.extend(fake_labels)

    all_images.extend(ru_images)
    all_labels.extend(ru_labels)

    all_images.extend(monty_images)
    all_labels.extend(monty_labels)
    train = pd.DataFrame({'images': all_images, 'labels': all_labels})

    with open('data.pkl', 'wb') as f:
        pickle.dump((train, valid_images, valid_labels), f)
else:
    print('using cached train valid data')
    with open('data.pkl', 'rb') as f:
        train, valid_images, valid_labels = pickle.load(f)

print(Counter(train.labels.values))
print(Counter(valid_labels))


class DeepFakeModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = get_model(name=args.model)
        self.val_preds = []
        self.val_labels = []

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x.float()
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}

        y_cpu = y.to('cpu')
        y_hat_cpu = y_hat.to('cpu')

        try:
            tensorboard_logs['trn/roc_auc'] = roc_auc_score(
                y_cpu.numpy(),
                torch.softmax(y_hat_cpu, dim=1).cpu().numpy()[:, 1]
            )
            tensorboard_logs['trn/f1'] = f1_score(
                y_cpu.numpy(),
                torch.softmax(y_hat_cpu, dim=1).cpu().numpy()[:, 1]
            )
        except:
            tensorboard_logs['trn/roc_auc'] = 0

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x = x.float()
        y_hat = self.forward(x)

        y_cpu = y.to('cpu')
        y_hat_cpu = y_hat.to('cpu')

        self.val_labels.append(y_cpu.numpy())
        self.val_preds.append(torch.softmax(y_hat_cpu, dim=1).numpy())

        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        preds = np.concatenate(self.val_preds)
        labels = np.concatenate(self.val_labels)

        self.val_labels = []
        self.val_preds = []
        try:
            score = roc_auc_score(labels, preds[:, 1])
        except:
            score = 0

        tensorboard_logs['val_roc_auc'] = score

        return {'avg_val_loss': avg_loss,
                'log': tensorboard_logs,
                }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def train_dataloader(self):
        train_ds = ImageListDs(images=train.images.values, labels=train.labels.values, aug=train_aug)
        sampler = RandomSampler(data_source=train_ds, replacement=True, num_samples=int(len(train_ds) / 10.0))
        train_loader = DataLoader(train_ds, num_workers=12, batch_size=BATCH_SIZE, sampler=sampler)
        return train_loader

    def val_dataloader(self):
        valid_ds = ImageListDs(images=valid_images, labels=valid_labels, aug=valid_aug)
        valid_loader = DataLoader(valid_ds, shuffle=False, num_workers=12, batch_size=BATCH_SIZE)
        return valid_loader


torch.multiprocessing.freeze_support()
deep_fake_module = DeepFakeModule()

trainer = pl.Trainer(gpus=torch.cuda.device_count(), max_epochs=30, )
trainer.fit(deep_fake_module)
