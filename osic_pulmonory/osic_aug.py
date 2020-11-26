from albumentations import Compose, Normalize, Resize, CLAHE, ShiftScaleRotate, RandomResizedCrop
from albumentations.pytorch import ToTensorV2


def get_osic_train_aug(name):
    return Compose([
        Resize(width=400, height=400, always_apply=True),
        RandomResizedCrop(width=384, height=384, scale=(0.5, 0.9), ratio=(0.5, 1), always_apply=True),
        CLAHE(),
        ShiftScaleRotate(),
        Normalize(mean=0.5, std=0.5, always_apply=True),
        ToTensorV2(always_apply=True),
    ], p=0.3)


def get_osic_valid_aug(name):
    return Compose([
        Resize(width=384, height=384, always_apply=True),
        Normalize(mean=0.5, std=0.5, always_apply=True),
        ToTensorV2(always_apply=True),
    ])
