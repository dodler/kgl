from albumentations import Compose, Resize, RandomCrop, Flip, HorizontalFlip, VerticalFlip, Transpose, RandomRotate90, \
    ShiftScaleRotate, OneOf, Blur, MotionBlur, MedianBlur, GaussianBlur
from albumentations.pytorch import ToTensor

train_aug = Compose([
    RandomCrop(height=96, width=96, p=0.2),
    OneOf([
        VerticalFlip(p=0.2),
        HorizontalFlip(p=0.3),
        Transpose(p=0.2),
        RandomRotate90(p=0.2),
    ], p=0.3),
    ShiftScaleRotate(p=0.2),
    OneOf([
        Blur(p=0.2),
        MotionBlur(p=0.2),
        MedianBlur(p=0.2),
        GaussianBlur(p=0.2),
    ],p=0.3),
    Resize(196, 196, always_apply=True),
    ToTensor()
])

valid_aug = Compose([
    Resize(196, 196, always_apply=True),
    ToTensor()
])
