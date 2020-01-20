from albumentations import Compose, Resize, HorizontalFlip
from albumentations.pytorch import ToTensor

train_aug = Compose([
    HorizontalFlip(p=0.3),
    Resize(128, 128, always_apply=True),
    ToTensor()
])

valid_aug = Compose([
    Resize(128, 128, always_apply=True),
    ToTensor()
])
