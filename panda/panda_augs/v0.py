from albumentations import (
    RandomRotate90,
    Transpose, Flip, Compose,
    Resize, Normalize)
from albumentations.pytorch import ToTensor

SIZE = 320

p = 0.4
train_aug = Compose([
    RandomRotate90(),
    Flip(),
    Transpose(),
    Resize(width=SIZE, height=SIZE, always_apply=True),
    ToTensor(normalize={'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]})
], p=p)

valid_aug = Compose([
    Resize(width=SIZE, height=SIZE, always_apply=True),
    ToTensor(normalize={'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}),
], p=1.0)
