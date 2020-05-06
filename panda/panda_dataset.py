import os.path as osp

import cv2

TARGET_DIR = '/var/ssd_1t/kaggle_panda/'


class PandaImageDataset:
    def __init__(self, df, aug):
        self.df = df
        self.aug = aug

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = osp.join(TARGET_DIR, 'train_images_2048', self.df.iloc[idx, 0] + '.tiff-0.png')
        mask_path = osp.join(TARGET_DIR, 'train_label_masks_crop_2048', self.df.iloc[idx, 0] + '_mask.tiff-0.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        augmented = self.aug(image=img)  # , mask=mask)
        img = augmented['image']
        # mask = augmented['mask']

        gleason_score = self.df.iloc[idx, 2]

        return img, gleason_score
