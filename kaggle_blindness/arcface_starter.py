"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import

import os.path as osp

import albumentations as A
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils import model_zoo
from tqdm import *

from kaggle_blindness.model import AptosDS, ArcSEResnext50
from kaggle_blindness.train_utils import FocalLoss, validate, train, upd_lr

tqdm=tqdm_notebook



WORKERS = 2
CHANNEL = 3

import warnings
warnings.filterwarnings("ignore")
SIZE = 300
NUM_CLASSES = 5

df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

x = df_train['id_code'].values
y = df_train['diagnosis'].values

le=LabelEncoder()
y=le.fit_transform(y)

train_x, test_x, train_y, test_y=train_test_split(x,y, test_size=0.1)

train_x=[osp.join('../input/aptos2019-blindness-detection/train_images/',k+'.png') for k in train_x]
test_x=[osp.join('../input/aptos2019-blindness-detection/train_images/',k+'.png') for k in test_x]

train_aug= A.Compose([
    A.Resize(width=512, height=512),
    A.HorizontalFlip(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    A.GridDistortion(),
    ])

test_aug= A.Resize(width=512, height=512)

train_ds=AptosDS(train_x, train_y, train_aug)
test_ds=AptosDS(test_x, test_y, test_aug)

train_loader=torch.utils.data.DataLoader(train_ds, num_workers=4, batch_size=16, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_ds,  num_workers=4, batch_size=16, shuffle=False)

DEVICE=0
model=ArcSEResnext50(5)
model.to(DEVICE)

for param in model.parameters():
    param.requires_grad = True

# opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
# crit=torch.nn.CrossEntropyLoss()
crit=FocalLoss()

print(validate(model, test_loader))
for i in range(10):
    train(i, train_loader, model, opt, crit=crit, use_arc_metric=True)
    if i % 5 == 0:
        validate(model, test_loader, crit)
upd_lr(opt, 1e-4)

for i in range(10):
    train(i, train_loader, model, opt, crit=crit, use_arc_metric=True)
    if i % 5 == 0:
        validate(model, test_loader, crit)