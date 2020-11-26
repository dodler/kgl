from albumentations import Compose, Normalize, Resize, CLAHE, ShiftScaleRotate, RandomResizedCrop
from albumentations.pytorch import ToTensorV2


def get_rsna_train_aug(name):
    return Compose([
        CLAHE(),
        ShiftScaleRotate(),
        Normalize(always_apply=True),
        ToTensorV2(always_apply=True),
    ], p=0.1)


def get_rsna_valid_aug(name):
    return Compose([
        Normalize(always_apply=True),
        ToTensorV2(always_apply=True),
    ])
