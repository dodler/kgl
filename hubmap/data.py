import os

import cv2
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import numpy as np

bs = 64
nfolds = 4
fold = 0
SEED = 2020
LABELS = 'input/train.csv'
NUM_WORKERS = 4

# https://www.kaggle.com/iafoss/256x256-images
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    def __init__(self, fold=fold, train=True, tfms=None,
                 train_path='input/crops256/train/',
                 mask_path='input/crops256/masks/'):

        ids = pd.read_csv(LABELS).id.values
        kf = KFold(n_splits=nfolds, random_state=SEED, shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])

        self.mask_path = mask_path
        self.train_path = train_path
        self.fnames = [fname for fname in os.listdir(train_path) if fname.split('_')[0] in ids]
        self.train = train
        self.tfms = tfms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.train_path, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_path, fname), cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        return img2tensor((img / 255.0 - mean) / std), img2tensor(mask)
