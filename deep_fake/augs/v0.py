from albumentations import (
    RandomRotate90,
    Transpose, Flip, Compose,
    Resize, Normalize)
from albumentations.pytorch import ToTensor

p = 0.4
train_aug = Compose([
    RandomRotate90(),
    Flip(),
    Transpose(),
    Resize(width=224, height=224, always_apply=True),
    Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    ),
    ToTensor()
], p=p)

valid_aug = Compose([
    Resize(width=224, height=224, always_apply=True),
    Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    ),
    ToTensor()
], p=p)

