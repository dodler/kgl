import numpy as np
import pandas as pd
import cv2
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR, CyclicLR, \
    CosineAnnealingWarmRestarts

from bangali_19.beng_augs import train_aug_v0, valid_aug_v0
from bangali_19.beng_data import BengaliDataset
from bangali_19.beng_heads import HeadV1, Head, HeadV2

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
                          valid_aug=valid_aug_v0,
                          isfoss_norm=False,
                          channel_num=1):
    folds = pd.read_csv(folds_path)

    train_ids = folds[folds.fold != fold].values
    valid_ids = folds[folds.fold == fold].values

    train_dataset = BengaliDataset(path=img_path, values=train_ids, aug=train_aug, isfoss_norm=isfoss_norm, channel_num=channel_num)
    valid_dataset = BengaliDataset(path=img_path, values=valid_ids, aug=valid_aug, isfoss_norm=isfoss_norm, channel_num=channel_num)

    return train_dataset, valid_dataset


def get_dict_value_or_default(dict_, key, default_value):
    if key in dict_:
        return dict_[key]
    else:
        return default_value


def make_scheduler_from_config(optimizer, config):
    if 'schedule' in config:
        if config['schedule'] == 'reduce_lr_on_plateau':
            return ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
        elif config['schedule'] == 'cosine_annealing_warm_restarts':
            T_0 = get_dict_value_or_default(dict_=config, key='T_0', default_value=4)
            return CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0)
        elif config['schedule'] == 'cosine_annealing':
            return CosineAnnealingLR(optimizer, T_max=4)
        elif config['schedule'] == 'exponential':
            return ExponentialLR(optimizer, gamma=0.99)
        elif config['schedule'] == 'cyclic':

            max_lr = get_dict_value_or_default(config, 'max_lr', 1e-1)
            base_lr = get_dict_value_or_default(config, 'base_lr', 1e-4)
            step_size_down = get_dict_value_or_default(config, 'step_size_down', 2000)
            mode = get_dict_value_or_default(config, 'cycle_mode', 'triangular')

            return CyclicLR(optimizer,
                            base_lr=base_lr,
                            max_lr=max_lr,
                            step_size_down=step_size_down,
                            mode=mode)
        raise Exception('check your config, config not supported')
    else:
        return ReduceLROnPlateau(optimizer, factor=0.1, patience=5)


def get_head_cls(head):
    if head == 'V1':
        return HeadV1
    elif head == 'V2':
        return HeadV2
    else:
        raise Exception('head ' + str(head) + ' is not supported')


def get_head(isfoss_head, head, in_size):
    if head != 'V0':
        head_cls = get_head_cls(head)
        return head_cls(in_size, 168), \
               head_cls(in_size, 11), \
               head_cls(in_size, 7)
    if isfoss_head:
        return Head(in_size, 168), \
               Head(in_size, 11), \
               Head(in_size, 7)
    else:
        return nn.Linear(in_size, 168), \
               nn.Linear(in_size, 11), \
               nn.Linear(in_size, 7)


if __name__ == '__main__':
    train_ds, _ = bengali_ds_from_folds()
    print(next(iter(train_ds)))
