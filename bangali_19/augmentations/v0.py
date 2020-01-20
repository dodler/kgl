from albumentations import Compose, Resize, RandomCrop, Flip, HorizontalFlip
from albumentations.pytorch import ToTensor

train_aug = Compose([
    RandomCrop(height=96, width=96, p=0.2),
    HorizontalFlip(p=0.3),
    Resize(128, 128, always_apply=True),
    ToTensor()
])

valid_aug = Compose([
    Resize(128, 128, always_apply=True),
    ToTensor()
])
