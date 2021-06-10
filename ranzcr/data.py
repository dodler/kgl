import cv2
import numpy as np


class RanzcrDs:
    def __init__(self, df, aug, path, logits_path=None):
        self.df = df
        self.aug = aug
        self.path = path
        self.cols = np.array(['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                              'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged',
                              'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline',
                              'CVC - Normal', 'Swan Ganz Catheter Present'], dtype=str)

        if logits_path is not None:
            self.logits = np.load(logits_path)
        else:
            self.logits = None

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        if self.logits is not None:
            dist_logits = self.logits[idx]
        else:
            dist_logits = None

        img_idx = self.df.StudyInstanceUID.values[idx]
        img_path = '{}/{}.jpg'.format(self.path, img_idx)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.aug(image=img)['image']

        labels = self.df[self.cols].values[idx]
        if dist_logits is not None:
            return img, labels, idx, dist_logits
        else:
            return img, labels, idx
