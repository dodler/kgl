import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize


class IntracranialDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):

        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels
        self.norm = Normalize([0.051267407775610244, 0.051267407775610244, 0.051267407775610244],
                              [0.09457648820407062, 0.09457648820407062, 0.09457648820407062])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.png')
        img = cv2.imread(img_name)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = self.norm(img)

        if self.labels:

            labels = torch.tensor(
                self.data.loc[
                    idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}

        else:

            return {'image': img}