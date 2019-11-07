import os

import cv2
from albumentations.pytorch import ToTensor

from clouds_sat.cloud_utils import make_mask, get_training_augmentation, \
    get_preprocessing, get_validation_augmentation

import segmentation_models_pytorch as smp

train_on_gpu = True

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

import albumentations as albu

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


class CloudDataset(Dataset):
    def __init__(self, path, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
                 transforms=albu.Compose([albu.HorizontalFlip(), ToTensor()]),
                 preprocessing=None):
        self.df = df
        if datatype != 'test':
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)


class CloudDatasetSegAndCls(Dataset):
    def __init__(self, path, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
                 transforms=albu.Compose([albu.HorizontalFlip(), ToTensor()]),
                 preprocessing=None):
        self.df = df
        self.df['HavingDefection'] = self.df['EncodedPixels'].map(lambda x: 0 if x is np.nan else 1)
        self.all_labels = np.array(df['HavingDefection']).reshape(-1, 4)

        if datatype != 'test':
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        label = self.all_labels[idx]
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return {"seg_features": img, "seg_targets": mask.astype(np.float32), "cls_targets": label.astype(np.float32)}

    def __len__(self):
        return len(self.img_ids)


def ds_from_folds(path, folds_path='stage_1_train_folds.csv', fold=0,
                  train_aug=get_training_augmentation(),
                  valid_aug=get_validation_augmentation()):
    folds = pd.read_csv(folds_path)

    main_col = 'Image_Label'
    train = pd.read_csv(f'{path}/train.csv')
    train['label'] = train[main_col].apply(lambda x: x.split('_')[1])
    train['im_id'] = train[main_col].apply(lambda x: x.split('_')[0])

    train_ids = folds[folds.fold != fold]['img_id'].values
    valid_ids = folds[folds.fold == fold]['img_id'].values

    train_dataset = CloudDataset(path=path, df=train, datatype='train',
                                 img_ids=train_ids,
                                 transforms=train_aug,
                                 preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(path=path, df=train,
                                 datatype='valid',
                                 img_ids=valid_ids,
                                 transforms=valid_aug,
                                 preprocessing=get_preprocessing(preprocessing_fn))

    return train_dataset, valid_dataset


def ds2heads_from_folds(path, folds_path='stage_1_train_folds.csv', fold=0,
                        train_aug=get_training_augmentation(),
                        valid_aug=get_validation_augmentation()):
    folds = pd.read_csv(folds_path)

    main_col = 'Image_Label'
    train = pd.read_csv(f'{path}/train.csv')
    train['label'] = train[main_col].apply(lambda x: x.split('_')[1])
    train['im_id'] = train[main_col].apply(lambda x: x.split('_')[0])

    train_ids = folds[folds.fold != fold]['img_id'].values
    valid_ids = folds[folds.fold == fold]['img_id'].values

    train_dataset = CloudDatasetSegAndCls(path=path, df=train, datatype='train',
                                          img_ids=train_ids,
                                          transforms=train_aug,
                                          preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDatasetSegAndCls(path=path, df=train,
                                          datatype='valid',
                                          img_ids=valid_ids,
                                          transforms=valid_aug,
                                          preprocessing=get_preprocessing(preprocessing_fn))

    return train_dataset, valid_dataset
