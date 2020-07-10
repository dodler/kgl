
from albumentations import RandomResizedCrop, Compose, Flip, Resize, Normalize
from albumentations.pytorch import ToTensorV2, ToTensor

SIZE = 224

train_aug = Compose([
    RandomResizedCrop(height=SIZE, width=SIZE, ),
    Flip(p=0.3),
    Normalize(),
    ToTensorV2(),
])

valid_aug = Compose([
    Resize(width=SIZE, height=SIZE),
    Normalize(),
    ToTensorV2(),
])