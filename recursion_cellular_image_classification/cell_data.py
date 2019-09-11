import sys

sys.path.append('/home/lyan/Documents/enorm/enorm')

PRINT_FREQ = 100

sys.path.append('/home/lyan/Documents/rxrx1-utils')

import torchvision.transforms as transforms
import pandas as pd
import cv2
import numpy as np

import torch


# from enorm import ENorm


class ImagesDS():
    def __init__(self, csv_file, img_dir, mode='train', site=1,
                 channels=[1, 2, 3, 4, 5, 6],
                 aug=None,
                 target_size=512):

        df = pd.read_csv(csv_file)
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.aug = aug
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.485, 0.456, 0.456, 0.406, 0.406],
                                              std=[0.229, 0.229, 0.224, 0.224, 0.225, 0.225])
        self.target_size=target_size

    def _get_img_path(self, index, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir, self.mode, experiment, f'Plate{plate}', f'{well}_s{self.site}_w{channel}.png'])

    def _load_img_as_tensor(self, file_name):
        return self.to_tensor(cv2.imread(file_name, cv2.IMREAD_GRAYSCALE))

    def __getitem__(self, index):
        # paths = [self._get_img_path(index, ch) for ch in self.channels]
        # img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])

        paths = [self._get_img_path(index, ch) for ch in self.channels]

        target = {}
        for i in range(len(paths)):
            if i == 0:
                prefix = 'image'
            else:
                prefix = 'image' + str(i)
            img = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
            if self.target_size != 512:
                img=cv2.resize(img, (self.target_size, self.target_size))
            target[prefix] = img

        augmented = self.aug(**target)
        img = np.zeros((self.target_size, self.target_size, 6), dtype=np.uint8)
        k = list(augmented.keys())
        for i in range(len(augmented.keys())):
            img[:, :, i] = augmented[k[i]]

        img=self.to_tensor(img)

        if self.mode == 'train':
            return img, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len
