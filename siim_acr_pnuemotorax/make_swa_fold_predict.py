import segmentation_models_pytorch as smp
import torch
import pickle
import numpy as np
import pandas as pd
from compress_pickle import dump, load
import matplotlib.pyplot as plt
import random

from tqdm import *
tqdm=tqdm_notebook

import cv2
import numpy as np

import os
import os.path as osp

import torchvision.transforms as transforms

import os
import os.path as osp

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import sys
sys.path.append('/home/lyan/Documents/kaggle/')
from segmentation.custom_unet import Unet

ENCODER = 'efficientnet-b1'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

CLASSES = ['pneumo']
ACTIVATION = 'sigmoid'

model = Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

to_tensor = transforms.ToTensor()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

checkpoints_list=[
    'efficientnet-b1_unet_Adamw_6_bce-dice_fold0__swa_2019-08-03_14_08_03/epoch_-1.pth',
    'efficientnet-b1_unet_Adamw_6_bce-dice_fold1__swa_2019-08-03_16_36_49/epoch_-1.pth',
    'efficientnet-b1_unet_Adamw_6_bce-dice_fold2__swa_2019-08-03_18_13_24/epoch_-1.pth',
    'efficientnet-b1_unet_Adamw_6_bce-dice_fold3__swa_2019-08-03_19_59_36/epoch_-1.pth',
    'efficientnet-b1_unet_Adamw_6_bce-dice_fold4__swa_2019-08-03_22_14_56/epoch_-1.pth'
]

checkpoints_list=[osp.join('/var/data/checkpoints/',k) for k in checkpoints_list]

model.eval()
model.to(0)

non_empty_masks_path='/var/ssd_1t/siim_acr_pneumo/stuff_annotations_trainval2017/annotations/masks_non_empty'
img_path='/var/ssd_1t/siim_acr_pneumo/val2017'
img_ids=os.listdir(img_path)

masks = []
imgs = []
for im in tqdm(img_ids):

    if not osp.exists(osp.join(non_empty_masks_path, im)):
        continue

    m = cv2.imread(osp.join(non_empty_masks_path, im), cv2.IMREAD_GRAYSCALE)
    masks.append(m)

    img = cv2.imread(osp.join(img_path, im))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imgs.append(img)