import cv2
from torchvision.transforms import Normalize
import pydicom
from rsna_hemorr.hem_augs import transform_train, transform_test, transform_train_raw, transform_test_raw
from skimage import exposure

dir_csv = '/var/ssd_1t/rsna_intra_hemorr/'
dir_train_img_png = '/var/ssd_1t/rsna_intra_hemorr/stage_1_train_png_224x/'
dir_test_img_png = '/var/ssd_1t/rsna_intra_hemorr/stage_1_test_png_224x/'

dir_train_img_dcm = '/var/ssd_1t/rsna_intra_hemorr/stage_1_train_images/'
dir_test_img_dcm = '/var/ssd_1t/rsna_intra_hemorr/stage_1_test_images/'

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IntracranialDataset(Dataset):

    def __init__(self, df, path, labels, transform=None):

        self.path = path
        self.data = df
        print('data len', self.data.shape)
        self.transform = transform
        self.labels = labels
        self.norm = Normalize([0.051267407775610244, 0.051267407775610244, 0.051267407775610244],
                              [0.09457648820407062, 0.09457648820407062, 0.09457648820407062])

        if self.labels:
            undersample_seed = 0
            print(self.data["any"].value_counts())
            num_ill_patients = self.data[self.data["any"] == 1].shape[0]
            print(num_ill_patients)

            healthy_patients = self.data[self.data["any"] == 0].index.values
            healthy_patients_selection = np.random.RandomState(undersample_seed).choice(
                healthy_patients, size=num_ill_patients * 4, replace=False
            )
            len(healthy_patients_selection)

            sick_patients = self.data[self.data["any"] == 1].index.values
            selected_patients = list(set(healthy_patients_selection).union(set(sick_patients)))
            len(selected_patients) / 2

            self.data = self.data.loc[selected_patients].copy().reset_index().drop('index', axis=1)
            print(self.data["any"].value_counts())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.png')
        img = cv2.imread(img_name)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = self.norm(img)
        if self.labels:
            labels = torch.tensor(
                self.data.loc[
                    idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return img, labels.float()

        else:
            return img


def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def window_image(img, window_center, window_width, intercept, slope):
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img


class IntracranialDatasetRaw(Dataset):

    def __init__(self, df, path, labels, transform=None, do_norm=True):

        self.path = path
        self.data = df
        print('data len', self.data.shape)
        self.transform = transform
        self.labels = labels
        self.do_norm = do_norm
        self.norm = Normalize([0.6],
                              [0.25])

        print(self.data.head())
        # if self.labels:
        #     undersample_seed = 0
        #     print(self.data["any"].value_counts())
        #     num_ill_patients = self.data[self.data["any"] == 1].shape[0]
        #     print(num_ill_patients)
        #
        #     healthy_patients = self.data[self.data["any"] == 0].index.values
        #     healthy_patients_selection = np.random.RandomState(undersample_seed).choice(
        #         healthy_patients, size=num_ill_patients, replace=False
        #     )
        #     len(healthy_patients_selection)
        #
        #     sick_patients = self.data[self.data["any"] == 1].index.values
        #     selected_patients = list(set(healthy_patients_selection).union(set(sick_patients)))
        #     len(selected_patients) / 2
        #
        #     self.data = self.data.loc[selected_patients].copy().reset_index().drop('index', axis=1)
        #     print(self.data["any"].value_counts())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.dcm')
        data = pydicom.read_file(img_name)
        img = data.pixel_array

        img = exposure.equalize_hist(img) * 255
        img = img.reshape(img.shape[0], img.shape[1], 1).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        if self.do_norm:
            img = self.norm(img)

        if self.labels:

            labels = torch.tensor(
                self.data.loc[
                    idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            # return {'image': img, 'labels': labels}
            return img, labels.float()

        else:

            return img


def hem_png_from_folds(image_path=dir_train_img_png, folds_path='stage_1_train_folds.csv', fold=0):
    train = pd.read_csv(folds_path)

    png = glob.glob(os.path.join(image_path, '*.png'))
    png = [os.path.basename(png)[:-4] for png in png]
    png = np.array(png)

    train_df = train[train['Image'].isin(png)][train.fold != fold].reset_index().drop('index', axis=1)
    valid_df = train[train['Image'].isin(png)][train.fold == fold].reset_index().drop('index', axis=1)

    train_dataset = IntracranialDataset(
        df=train_df, path=image_path, transform=transform_train, labels=True)

    valid_dataset = IntracranialDataset(
        df=valid_df, path=image_path, transform=transform_test, labels=True)

    return train_dataset, valid_dataset


def hem_dcm_from_folds(folds_path='stage_1_train_folds.csv', fold=0):
    train = pd.read_csv(folds_path)

    dcm = glob.glob(os.path.join(dir_train_img_dcm, '*.dcm'))
    dcm = [os.path.basename(dcm)[:-4] for dcm in dcm]
    dcm = np.array(dcm)

    train_df = train[train['Image'].isin(dcm)][train.fold != fold].reset_index().drop('index', axis=1)
    valid_df = train[train['Image'].isin(dcm)][train.fold == fold].reset_index().drop('index', axis=1)

    train_dataset = IntracranialDatasetRaw(
        df=train_df, path=dir_train_img_dcm, transform=transform_train_raw, labels=True)

    valid_dataset = IntracranialDatasetRaw(
        df=valid_df, path=dir_train_img_dcm, transform=transform_test_raw, labels=True)

    return train_dataset, valid_dataset


if __name__ == '__main__':
    t, _ = hem_png_from_folds()
    values = next(iter(t))
    print('test output shape', values[0].shape)

    t, _ = hem_dcm_from_folds()
    values = next(iter(t))
    print(values[0].shape, values[0].mean(), values[0].max())
