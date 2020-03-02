from albumentations import Compose, Resize, RandomCrop, Flip, HorizontalFlip, VerticalFlip, Transpose, RandomRotate90, \
    ShiftScaleRotate, OneOf, Blur, MotionBlur, MedianBlur, GaussianBlur, RandomBrightness, RandomBrightnessContrast, \
    Normalize
from albumentations.pytorch import ToTensor

train_aug = Compose([
    Resize(128, 128, always_apply=True),
    Normalize(mean=0.06922848809290576, std=0.20515700083327537),
    ToTensor()
])

valid_aug = Compose([
    Resize(128, 128, always_apply=True),
    Normalize(mean=0.06922848809290576, std=0.20515700083327537),
    ToTensor()
])
