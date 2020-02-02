from albumentations import Compose, Resize, HorizontalFlip, CenterCrop
from albumentations.pytorch import ToTensor

train_aug = Compose([
    HorizontalFlip(p=0.3),
    Resize(128, 128, always_apply=True),
    ToTensor()
])

valid_aug = Compose([
    CenterCrop(96, 96, always_apply=True),
    Resize(128, 128, always_apply=True),
    ToTensor()
])
