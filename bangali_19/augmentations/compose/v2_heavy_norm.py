from albumentations import Compose, Resize, RandomCrop, Flip, HorizontalFlip, VerticalFlip, Transpose, RandomRotate90, \
    ShiftScaleRotate, OneOf, Blur, MotionBlur, MedianBlur, GaussianBlur, RandomBrightness, RandomBrightnessContrast, \
    Normalize
from albumentations.pytorch import ToTensor

train_aug = Compose([
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
    Normalize(mean=0.06922848809290576, std=0.20515700083327537),
    ToTensor()
])

valid_aug = Compose([
    Resize(128, 128, always_apply=True),
    Normalize(mean=0.06922848809290576, std=0.20515700083327537),
    ToTensor()
])
