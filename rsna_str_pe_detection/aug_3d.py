from monai.transforms import LoadNifti, Randomizable, apply_transform
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor, RandAffine
import numpy as np


def get_rsna_train_aug(name=None, image_size=160):
    return Compose([ScaleIntensity(),
             Resize((image_size, image_size, image_size)),
             RandAffine(
                 prob=0.5,
                 translate_range=(5, 5, 5),
                 rotate_range=(np.pi * 4, np.pi * 4, np.pi * 4),
                 scale_range=(0.15, 0.15, 0.15),
                 padding_mode='border'),
             ToTensor()])


def get_rsna_valid_aug(name=None, image_size=160):
    return Compose([ScaleIntensity(), Resize((image_size, image_size, image_size)), ToTensor()])
