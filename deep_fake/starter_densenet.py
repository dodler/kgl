import os
import os.path as osp
import random
from collections import Counter

import pandas as pd
import pretrainedmodels as pm
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from deep_fake.augs.v0 import train_aug, valid_aug
from deep_fake.deep_fake_data import ImageListDs

FAKE_LABEL = 1
TRUE_LABEL = 0

true_img_dir = '/var/ssd_1t/kaggle_deepfake/us_images_crops/'
false_img_dir = '/var/ssd_1t/nvidia_df_57g/deep_fake_nvidia_512/'

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

# train, valid = train_test_split(data, test_size=0.001)

print(Counter(train.labels.values))
print(Counter(valid_labels))

BATCH_SIZE = 42 * 2


class DeepFakeModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        model=pm.densenet121()
        model._fc = torch.nn.Linear(1536, 2)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x
        # return x.squeeze()

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x.float()
        # y = y.float()
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        x = x.float()
        # y = y.float()
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        train_ds = ImageListDs(images=train.images.values, labels=train.labels.values, aug=train_aug)
        # samler = DistributedSampler(train_ds)
        n = int(len(train_ds) / 10)
        sampler = RandomSampler(data_source=train_ds)
        train_loader = DataLoader(train_ds, shuffle=True, num_workers=12, batch_size=BATCH_SIZE)
        return train_loader

    def val_dataloader(self):
        valid_ds = ImageListDs(images=valid_images, labels=valid_labels, aug=valid_aug)
        valid_loader = DataLoader(valid_ds, shuffle=False, num_workers=12, batch_size=BATCH_SIZE)
        return valid_loader


torch.multiprocessing.freeze_support()
deep_fake_module = DeepFakeModule()

trainer = pl.Trainer(gpus=2, distributed_backend='dp', max_epochs=2)
trainer.fit(deep_fake_module)
