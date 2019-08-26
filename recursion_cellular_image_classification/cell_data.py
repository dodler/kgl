import sys

sys.path.append('/home/lyan/Documents/enorm/enorm')

PRINT_FREQ = 100

sys.path.append('/home/lyan/Documents/rxrx1-utils')

import pandas as pd
import cv2

import torch


# from enorm import ENorm


class ImagesDS():
    def __init__(self, csv_file, img_dir, mode='train', site=1, channels=[1, 2, 3, 4, 5, 6], aug=None):

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

    def _get_img_path(self, index, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir, self.mode, experiment, f'Plate{plate}', f'{well}_s{self.site}_w{channel}.png'])

    def _load_img_as_tensor(self, file_name):
        return self.to_tensor(cv2.imread(file_name, cv2.IMREAD_GRAYSCALE))

    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
        if self.mode == 'train':
            return img, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len
