import cv2
import numpy as np


class RanzcrDs:
    def __init__(self, df, aug, path):
        self.df = df
        self.aug = aug
        self.path = path
        self.cols = np.array(['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                              'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged',
                              'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline',
                              'CVC - Normal', 'Swan Ganz Catheter Present'], dtype=str)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_idx = self.df.StudyInstanceUID.values[idx]
        img_path = '{}/{}.jpg'.format(self.path, img_idx)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.aug(image=img)['image']

        labels = self.df[self.cols].values[idx]
        return img, labels, idx
