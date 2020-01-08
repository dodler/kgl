import numpy as np
import pandas as pd
import cv2

from bangali_19.beng_augs import train_aug_v0, valid_aug_v0
from bangali_19.beng_data import BengaliDataset

HEIGHT = 137
WIDTH = 236
SIZE = 128


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=SIZE, pad=16, width=WIDTH, height=HEIGHT,
                inter=cv2.INTER_LANCZOS4):
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < width - 13) else width
    ymax = ymax + 10 if (ymax < height - 10) else height
    img = img0[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax - xmin, ymax - ymin
    l = max(lx, ly) + pad
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode='constant')
    return cv2.resize(img, (size, size), interpolation=inter)


def bengali_ds_from_folds(img_path='/var/ssd_1t/kaggle_bengali/jpeg_crop/',
                          folds_path='/home/lyan/Documents/kaggle/bangali_19/folds.csv', fold=0,
                          train_aug=train_aug_v0,
                          valid_aug=valid_aug_v0):
    folds = pd.read_csv(folds_path)

    train_ids = folds[folds.fold != fold].values
    valid_ids = folds[folds.fold == fold].values

    train_dataset = BengaliDataset(path=img_path, values=train_ids, aug=train_aug)
    valid_dataset = BengaliDataset(path=img_path, values=valid_ids, aug=valid_aug)

    return train_dataset, valid_dataset


if __name__ == '__main__':
    train_ds, _ = bengali_ds_from_folds()
    print(next(iter(train_ds)))
