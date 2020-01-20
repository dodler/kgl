from albumentations import Compose, Resize, RandomCrop, Flip, HorizontalFlip, VerticalFlip, Transpose, RandomRotate90, \
    ShiftScaleRotate, OneOf, OpticalDistortion, HueSaturationValue
from albumentations.pytorch import ToTensor

train_aug = Compose([
    RandomCrop(height=96, width=96, p=0.2),
    OneOf([
        VerticalFlip(p=0.2),
        HorizontalFlip(p=0.3),
        Transpose(p=0.2),
        RandomRotate90(p=0.2),
    ],p=0.3),
    ShiftScaleRotate(p=0.2),
    HueSaturationValue(p=0.2),
    Resize(128, 128, always_apply=True),
    ToTensor()
])

valid_aug = Compose([
    Resize(128, 128, always_apply=True),
    ToTensor()
])
