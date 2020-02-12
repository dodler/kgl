import os.path as osp

import cv2


class BengaliDataset:
    def __init__(self, path, values, aug, isfoss_norm=False, channel_num=1):
        self.channel_num = channel_num
        self.isfoss_norm = isfoss_norm
        self.path = path
        self.aug = aug
        self.values = values

    def __len__(self):
        """return length of this dataset"""
        return self.values.shape[0]

    def __getitem__(self, item):
        img_id = self.values[item, 0]
        img_path = osp.join(self.path, img_id + '.png')
        if self.channel_num == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.aug(image=img)['image']
        if self.channel_num == 1:
            img = img.reshape(1, img.shape[0], img.shape[1]).float()
        return {
            'features': img,
            'h1_targets': int(self.values[item, 1]),
            'h2_targets': int(self.values[item, 2]),
            'h3_targets': int(self.values[item, 3]),
        }
