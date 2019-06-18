import pandas as pd
import os.path as osp
import os
import cv2
import torch
from sklearn.model_selection import train_test_split
from tqdm import *


class LandmarkDataset:
    def __init__(self, data_path, images_path, csv_file, aug, test=False):
        self.test = test
        self.aug = aug
        self.csv_file = csv_file
        self.images_path = images_path
        self.data_path = data_path

        data = pd.read_csv(osp.join(data_path, csv_file))
        if not test:
            print(data['landmark_id'].unique().shape)
        missing_data = []
        for i in tqdm(range(data.shape[0])):
            idx = data.iloc[i, 0]
            img_path = osp.join(images_path, str(idx) + '.jpg')
            if not osp.exists(img_path):
                missing_data.append(i)

        print('got data with shape', data.shape, 'head:', data.head())
        data.drop(missing_data, axis=0, inplace=True)
        print('data shape after missing drop', data.shape)
        if not test:
            data = data[data['landmark_id'] != 'None']
            print('dropped None landmark id')
        self.data = data

        if not test:
            assert 'landmark_id' in self.data.columns

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        idx = self.data.iloc[item, 0]
        img_path = idx + '.jpg'
        img_path = osp.join(self.images_path, img_path)

        # img = cv2.imread(img_path)
        img = read_turbo(img_path)
        img = img[:, :, (2, 1, 0)]  # BGR -> RGB

        if self.aug is not None:
            img = self.aug.augment_image(img)

        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        if not self.test:
            attr = int(self.data['landmark_id'].iloc[item])
            return img, torch.tensor(attr).long(), idx
        else:
            return img, idx


def make_data_split(csv_path, save_dir=None):
    data = pd.read_csv(csv_path)
    train, dev = train_test_split(data, test_size=0.05)
    train, val = train_test_split(train, test_size=0.05)

    if save_dir is None:
        save_dir = ""

    train.to_csv(osp.join(save_dir, 'train1.csv'), index=False)
    dev.to_csv(osp.join(save_dir, 'dev.csv'), index=False)
    val.to_csv(osp.join(save_dir, 'val.csv'), index=False)


if __name__ == '__main__':
    # ds = LandmarkDataset('/home/lyan/Documents/kaggle_data/google_landmark_retrieval/',
    #                      '/var/data/google_landmark_retrieval_train/', 'train.csv', None)
    # print(next(iter(ds)))
    #

    make_data_split('/home/lyan/Documents/kaggle_data/google_landmark_retrieval/train.csv', '/home/lyan/Documents/kaggle_data/google_landmark_retrieval/')