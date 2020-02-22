from albumentations import Compose, Resize, RandomCrop, Flip, HorizontalFlip, VerticalFlip, Transpose, RandomRotate90, \
    ShiftScaleRotate, OneOf, Blur, MotionBlur, MedianBlur, GaussianBlur, RandomBrightness, RandomBrightnessContrast
from albumentations.pytorch import ToTensor

train_aug = Compose([
    RandomCrop(height=96, width=96, p=0.2),
    OneOf([
        VerticalFlip(p=0.2),
        HorizontalFlip(p=0.3),
        Transpose(p=0.2),
        RandomRotate90(p=0.2),
    ], p=0.4),
    ShiftScaleRotate(p=0.4),
    OneOf([
        Blur(p=0.2),
        MotionBlur(p=0.2),
        MedianBlur(p=0.2),
        GaussianBlur(p=0.2),
    ], p=0.4),
    RandomBrightnessContrast(p=0.4,brightness_limit=0.4, contrast_limit=0.4),
    Resize(128, 128, always_apply=True),
    ToTensor()
])

valid_aug = Compose([
    Resize(128, 128, always_apply=True),
    ToTensor()
])
