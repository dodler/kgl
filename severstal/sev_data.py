import os
import os.path as osp
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms


class SegData():
    def __init__(self, img_ids, image_path, mask_path, aug=None):
        self.img_ids = img_ids
        self.aug = aug
        self.image_path = image_path
        self.mask_path = mask_path
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        img_name = osp.join(self.image_path, self.img_ids[item] + '.jpg')
        mask_name = osp.join(self.mask_path, self.img_ids[item] + '.png')
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        if self.aug is not None:
            augm = self.aug(image=img, mask=mask)

        img = augm['image']
        mask = augm['mask']

        mask = mask.astype(np.float32).reshape(1, mask.shape[0], mask.shape[1])

        return self.norm(self.to_tensor(img)), torch.from_numpy(mask)
