import os
import os.path as osp
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

class MultiClassData():
    def __init__(self, images_path, df):
        self.images_path = images_path
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        pass


class BinClassData():
    def __init__(self, images_path, df, aug=None, fold=0):
        self.aug = aug
        self.images_path = images_path
        self.df = df
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        img_p = osp.join(self.df.iloc[item, 0])
        img_p = img_p.split('_')[0]
        img=cv2.imread(img_p)

        if self.aug is not None:
            img=self.aug(image=img)

        return img, isinstance(self.df.iloc[item,1], str)



class SegClassData():
    def __init__(self, images_path, df):
        self.images_path = images_path
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        pass
