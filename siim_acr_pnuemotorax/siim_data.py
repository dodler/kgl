from __future__ import print_function

import os
import os.path as osp
import random

import pandas as pd
import cv2
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms

from segmentation.albs import aug_geom_color

kernel = np.ones((5, 5), np.uint8)


class SIIMDatasetSegmentation(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, aug, ext_img_ids=None):
        self.mask_dir = mask_dir
        self.image_dir = image_dir
        self.aug = aug

        if mask_dir is not None:
            img_ids = os.listdir(mask_dir)
        else:
            img_ids = os.listdir(image_dir)

        print(len(img_ids))

        if self.mask_dir is not None:
            all_exist_img_ids = []
            for im in img_ids:
                im_idx = im
                if osp.exists(osp.join(image_dir, im_idx)) and osp.exists(osp.join(mask_dir, im_idx)):
                    all_exist_img_ids.append(im_idx)
            self.img_ids = all_exist_img_ids
            print('all exists img ids len:', len(self.img_ids))
        else:
            self.img_ids = img_ids

        if ext_img_ids is not None:
            self.img_ids = ext_img_ids
            print('using img ids from args')

        self.height = 1024
        self.width = 1024
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        img_path = osp.join(self.image_dir, self.img_ids[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mask_dir is not None:
            mask = self.img_ids[idx]
            mask = osp.join(self.mask_dir, mask)
            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            # if random.random()>0.5 and mask.sum() > 15000: # value aggregated from train masks distribution
            #     kernel = np.ones((3, 3), np.uint8)
            #     # random dilation or erosion
            #     # for large masks to model inaccurate masking for large masks
            #     if random.random() > 0.5:
            #         mask = cv2.erode(mask, kernel, iterations=1)
            #     else:
            #         mask = cv2.dilate(mask, kernel, iterations=1)

        if self.aug is not None:
            if self.mask_dir is not None:
                augm = self.aug(image=img, mask=mask)
            else:
                augm = self.aug(image=img)
            img = augm['image']

            if self.mask_dir is not None:
                mask = augm['mask']
        mask = mask.astype(np.float32).reshape(1, mask.shape[0], mask.shape[1])

        if self.mask_dir is not None:
            return self.norm(self.to_tensor(img)), torch.from_numpy(mask)
        else:
            return self.norm(self.to_tensor(img))

    def __len__(self):
        return len(self.img_ids)


def from_folds(image_dir,
               mask_dir,
               aug_trn,
               aug_val,
               folds_path='/home/lyan/Documents/kaggle/siim_acr_pnuemotorax/folds.csv',
               fold=0):
    folds = pd.read_csv(folds_path)
    valid_ids_list = folds[folds.fold == fold].ids.values.tolist()
    train_ids_list = folds[folds.fold != fold].ids.values.tolist()

    valid_ids_list = [k.split('/')[-1] for k in valid_ids_list]
    train_ids_list = [k.split('/')[-1] for k in train_ids_list]

    trn_ds = SIIMDatasetSegmentation(image_dir=image_dir, mask_dir=mask_dir, aug=aug_trn)
    val_ds = SIIMDatasetSegmentation(image_dir=image_dir, mask_dir=mask_dir, aug=aug_val)

    trn_ds.img_ids = train_ids_list
    val_ds.img_ids = valid_ids_list

    return trn_ds, val_ds


def from_pickle(train_ids, holdout):
    pass


def from_sub(sub_path, test_mask_path, imgs_path, aug):
    sub = pd.read_csv(sub_path)
    ds = SIIMDatasetSegmentation(image_dir=imgs_path, mask_dir=test_mask_path, aug=aug)
    img_ids = []
    for v in sub.ImageId.values.tolist():
        if v != '-1':
            img_ids.append(v + '.png')
    return ds


if __name__ == '__main__':
    ds = SIIMDatasetSegmentation(image_dir='/var/ssd_1t/siim_acr_pneumo/train2017',
                                 mask_dir='/var/ssd_1t/siim_acr_pneumo/stuff_annotations_trainval2017/annotations/masks_non_empty/',
                                 aug=aug_geom_color)

    b = next(iter(ds))
    print(b[1].max())

    t = from_sub(
        sub_path='/home/lyan/Documents/kaggle/siim_acr_pnuemotorax/sub_blend.csv',
        test_mask_path='/var/ssd_1t/siim_acr_pneumo/test_masks',
        imgs_path='/var/ssd_1t/siim_acr_pneumo/test_png/',
        aug=None
    )

    for _ in t:
        pass
