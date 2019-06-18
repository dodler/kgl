import torch
from torchvision.transforms import Normalize, ToTensor
from torchvision.transforms import Compose as TCompose
import cv2
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import os.path as osp

from sklearn.preprocessing import LabelEncoder

from kaggle_lyan_utils import image_resize, read_turbo


def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility,
                                     'label_fields': ['category_id']})


class GoogleObjDetection:
    def __init__(self, base_dir, img_dir, tfm):
        self.aug = tfm
        self.img_dir = img_dir
        self.base_dir = base_dir

        d = pd.read_csv(osp.join(base_dir, 'meta', 'train-annotations-bbox.csv'))
        d['ImageIDINT'] = d.ImageID.apply(lambda x: int(x, 16))
        le = LabelEncoder()
        le.fit(d.LabelName.unique())
        d['labels_int'] = le.transform(d.LabelName)
        self.le = le
        self.d = d

        img_ids = [k[0:-4] for k in os.listdir(osp.join(base_dir, img_dir))]
        self.img_ids = [int(k, 16) for k in img_ids]

        self.imgs = [osp.join(base_dir, img_dir, k) for k in os.listdir(osp.join(base_dir, img_dir))]
        self.tfm = tfm

        self.norm = TCompose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        idx = self.img_ids[idx]
        img_meta = self.d[self.d.ImageIDINT == idx]
        img_fp = osp.join(self.base_dir, self.img_dir, img_meta.ImageID.values[0] + '.jpg')
        img=read_turbo(img_fp)
        # img = cv2.imread(img_fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = image_resize(img, width=256)

        box = []
        h, w, _ = img.shape
        for i in range(img_meta.shape[0]):
            xmin = img_meta.XMin.values[i]
            ymin = img_meta.YMin.values[i]
            xmax = img_meta.XMax.values[i]
            ymax = img_meta.YMax.values[i]

            box.append([xmin*w, ymin*h,
                        xmax*w, ymax*h])

        # box = np.array(box).astype(np.int)
        labels = img_meta.labels_int.values

        annotation = {'image': img, 'bboxes': box, 'category_id': labels}
        r = self.aug(**annotation)
        img = r['image']
        box = r['bboxes']
        labels = torch.as_tensor(r['category_id'], dtype=torch.int64)

        img = self.norm(img)
        box = torch.as_tensor(box, dtype=torch.float32)
        area = (box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0])
        iscrowd = torch.zeros((box.shape[0],), dtype=torch.int64)

        target = {"boxes": box, "labels": labels, "area": area,
                  "iscrowd": iscrowd}

        return img, target


if __name__ == '__main__':
    base_dir = '/var/ssd_1t/open_images_obj_detection/'
    aug = get_aug([VerticalFlip(p=0.5), RandomCrop(height=128, width=128)])
    ds = GoogleObjDetection(base_dir, 'train_small', aug)
    print(next(iter(ds)))
