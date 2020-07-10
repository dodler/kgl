import glob

import cv2
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from google_landmarks2.preproc import train_aug, valid_aug
from tqdm import tqdm


class GldData(torch.utils.data.Dataset):
    def __init__(self, df, aug, label_encoder):
        self.label_encoder = label_encoder
        self.aug = aug
        self.df = df
        images = glob.glob('{}/*/*/*/*'.format('/var/ssd_2t_1/kaggle_gld/train'))

        imgidx2path = {}
        for img in tqdm(images):
            img_name = img.split('/')[-1].split('.')[0]
            imgidx2path[img_name] = img
        self.imgidx2path = imgidx2path

        image_path2landmark = {}
        for i in tqdm(range(self.df.shape[0])):
            idx = self.df.iloc[i, 0]
            img_path = imgidx2path[idx]
            image_path2landmark[img_path] = self.df.iloc[i, 1]
        self.image_path2landmark = image_path2landmark

    def __getitem__(self, item):
        idx = self.df.iloc[item, 0]
        label = self.df.iloc[item, 1].reshape(1, -1)
        label = self.label_encoder.transform(label)[0]
        path = self.imgidx2path[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.aug(image=img)['image']
        return img, label

    def __len__(self):
        return self.df.shape[0]


if __name__ == '__main__':
    path = '/var/ssd_2t_1/kaggle_gld/'
    df = pd.read_csv('/home/lyan/Documents/kaggle/kaggle_landmarks/train_group_folds.csv')
    le = LabelEncoder()
    le.fit_transform(df.landmark_id)

    fold = 0
    train = df[df.fold != fold]
    valid = df[df.fold == fold]

    train_data = GldData(df=train, aug=train_aug, label_encoder=le)
    valid_data = GldData(df=valid, aug=valid_aug, label_encoder=le)

    it = iter(train_data)
    img, label = next(it)
    print(label)
    print(img.shape, label.shape)
