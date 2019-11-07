import os.path as osp

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


class SegData():
    def __init__(self, img_ids, image_dir, mask_dir, aug=None, progressive=False):
        self.progressive = progressive
        self.img_ids = img_ids
        self.aug = aug
        self.image_path = image_dir
        self.mask_path = mask_dir
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.34224, 0.34224, 0.34224], std=[0.13732, 0.13732, 0.13732])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, item):
        img_name = osp.join(self.image_path, self.img_ids[item] + '.jpg')
        mask_name = osp.join(self.mask_path, self.img_ids[item] + '.png')

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
        ch = 4

        if self.aug is not None:
            augm = self.aug(image=img, mask=mask)
            img = augm['image']
            mask = augm['mask']

        mask = mask.astype(np.float32).reshape(ch, mask.shape[0], mask.shape[1])

        return self.norm(self.to_tensor(img)), torch.from_numpy(mask)


class SevPretrain:
    def __init__(self, img_ids, image_dir, aug=None):
        self.img_ids = img_ids
        self.aug = aug
        self.image_path = image_dir
        self.labels = np.random.randint(0, 20, size=img_ids.shape[0])

        self.to_tensor = transforms.ToTensor()
        # self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm = transforms.Normalize(mean=[0.34224, 0.34224, 0.34224], std=[0.13732, 0.13732, 0.13732])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, item):
        img_name = osp.join(self.image_path, self.img_ids[item])

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        if self.aug is not None:
            augm = self.aug(image=img)
            img = augm['image']

        return self.norm(self.to_tensor(img)), self.labels[item]


class SevClass:
    def __init__(self, df, image_dir, aug=None):
        self.image_dir = image_dir
        self.aug = aug
        self.to_tensor = transforms.ToTensor()
        # self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm = transforms.Normalize(mean=[0.34224, 0.34224, 0.34224], std=[0.13732, 0.13732, 0.13732])

        n = int(df.shape[0] / 4)
        one_hots = np.zeros((n, 4), np.uint8)
        img = []
        for i in range(int(df.shape[0] / 4)):

            img.append(df.iloc[i * 4,0].split('_')[0])

            enc = df.iloc[i * 4:(i + 1) * 4].EncodedPixels.values
            for j in range(4):
                if isinstance(enc[j], str):
                    one_hots[i, j] = 1
        self.n = n
        self.one_hots = one_hots
        self.imgs = img

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        img = self.imgs[item]

        img_name = osp.join(self.image_dir, img)

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.aug is not None:
            augm = self.aug(image=img)
            img = augm['image']

        return self.norm(self.to_tensor(img)).float(), \
               torch.from_numpy(self.one_hots[item, :]).float()
