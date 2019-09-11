import json
import os
import os.path as osp

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from tqdm import *

from siim_acr_pnuemotorax.prediction_utils import sigmoid
from segmentation.segmentation import Unet

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from compress_pickle import dump

to_tensor = transforms.ToTensor()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def from_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config


config = from_config('fold_predict_configs/efficientnet-b0_v4.json')

checkpoints_list = config['checkpoints']
checkpoints_list = [osp.join(config['base_dir'], k) for k in checkpoints_list]

print(config['comment'])

ENCODER = config['backbone']
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

model.eval()
model.to(0)


def simple_predict(model_ft, raw_img):
    img = to_tensor(raw_img)
    img = norm(img).cuda().unsqueeze(0)
    with torch.no_grad():
        prd = model_ft(img)
        return torch.sigmoid(prd).detach().cpu().numpy()


def tta(model_ft, raw_img):
    result = simple_predict(model_ft, raw_img)
    pred = simple_predict(model_ft, cv2.flip(raw_img, 0))
    result += cv2.flip(pred.reshape(1024, 1024), 0).reshape(1, 1, 1024, 1024)

    pred = simple_predict(model_ft, cv2.flip(raw_img, 1))
    result += cv2.flip(pred.reshape(1024, 1024), 0).reshape(1, 1, 1024, 1024)

    return result / 3.0


# simple_predict = tta


def make_predict(model_ft, target_imgs):
    result = np.zeros((len(target_imgs), 1024, 1024)).astype(np.float32)
    for i in tqdm(range(len(target_imgs))):
        result[i] = simple_predict(model_ft, target_imgs[i])
    return result


def get_holdout_imgs(
        non_empty_masks_path='/var/ssd_1t/siim_acr_pneumo/stuff_annotations_trainval2017/annotations/masks_non_empty',
        img_path='/var/ssd_1t/siim_acr_pneumo/holdout/'
):
    img_ids = os.listdir(img_path)

    result_masks = []
    result_imgs = []
    for im in tqdm(img_ids):

        if not osp.exists(osp.join(non_empty_masks_path, im)):
            continue

        m = cv2.imread(osp.join(non_empty_masks_path, im), cv2.IMREAD_GRAYSCALE)
        result_masks.append(m)

        img = cv2.imread(osp.join(img_path, im))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result_imgs.append(img)
    return result_imgs, result_masks


def get_test_imgs(sub_path='subf93.csv'):
    sub = pd.read_csv(sub_path)
    img_path = '/var/ssd_1t/siim_acr_pneumo/test_png'
    result_imgs = []
    for i in tqdm(range(sub.shape[0])):
        img = cv2.imread(osp.join(img_path, sub.iloc[i, 0] + '.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result_imgs.append(img)

    return result_imgs


val_imgs, masks = get_holdout_imgs()

result = np.zeros((len(val_imgs), 1024, 1024)).astype(np.float32)
for ckpt in checkpoints_list:
    ckpt_torch = torch.load(ckpt)
    print('ckpt', ckpt, 'score', str(ckpt_torch['score']))
    model.load_state_dict(ckpt_torch['net'])
    result += make_predict(model, val_imgs)

print('dumped val result to ', 'val_' + checkpoints_list[0].split('/')[-2] + '.gz')
dump(result.astype(np.float16), 'val_' + checkpoints_list[0].split('/')[-2] + '.gz')

test_imgs = get_test_imgs()
result = np.zeros((len(test_imgs), 1024, 1024)).astype(np.float32)
for ckpt in checkpoints_list:
    model.load_state_dict(torch.load(ckpt)['net'])
    result += make_predict(model, test_imgs)

print('dumped test result to ', 'test_' + checkpoints_list[0].split('/')[-2] + '.gz')
dump(result.astype(np.float16), 'test_' + checkpoints_list[0].split('/')[-2] + '.gz')
