import os

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import Resize, Normalize, Compose, RandomResizedCrop, CenterCrop, CLAHE, ShiftScaleRotate, \
    RandomCrop
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class MelanomaDataset(Dataset):

    @staticmethod
    def get_meta_features():
        return ['sex',
                'age_approx',
                'site_anterior torso',
                'site_head/neck',
                'site_lateral torso',
                'site_lower extremity',
                'site_oral/genital',
                'site_palms/soles',
                'site_posterior torso',
                'site_torso',
                'site_upper extremity',
                'site_nan']

    def __init__(self, df, path, tfm, meta_features=None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            path (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age

        """
        self.df = df
        self.path = path
        self.tfm = tfm
        self.meta_features = meta_features

    def __getitem__(self, index):
        im_path = os.path.join(self.path, self.df.iloc[index]['image_id'] + '.jpg')
        x = cv2.imread(im_path)
        meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)

        x = self.tfm(image=x)['image']

        y = self.df.iloc[index]['target']
        return (x, meta), y

    def __len__(self):
        return self.df.shape[0]


if __name__ == '__main__':
    tfm = Compose([
        # RandomResizedCrop(width=384, height=384, scale=(0.5, 0.9), ratio=(0.5, 1)),
        Resize(width=400, height=400, always_apply=True),
        RandomCrop(width=384, height=384, always_apply=True),
        CLAHE(),
        ShiftScaleRotate(),
        Normalize(),
        ToTensorV2(),
    ], p=0.3)

    df = pd.read_csv('/home/lyan/Documents/kaggle/isic_melanoma/group_fold_train_512.csv')
    path = '/var/ssd_2t_1/kaggle_isic/ds_512_2/512x512-dataset-melanoma/512x512-dataset-melanoma'
    ds = MelanomaDataset(df=df, path=path, tfm=tfm, meta_features=MelanomaDataset.get_meta_features())
    dl = torch.utils.data.DataLoader(dataset=ds, shuffle=False, num_workers=4, batch_size=24)
    for (x, meta), y in dl:
        print(x.shape)
        print(meta.shape)
        print(y.shape)
        break
