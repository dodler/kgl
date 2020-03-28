from albumentations import (
    RandomRotate90,
    Transpose, Flip, Compose,
    Resize, Normalize, RandomGridShuffle)
from albumentations.pytorch import ToTensor

p = 0.4

augs = {
    'wgrid': [
        Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            RandomGridShuffle(),
            Resize(width=224, height=224, always_apply=True),
            ToTensor(normalize={'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]})
        ], p=p),
        Compose([
            Resize(width=224, height=224, always_apply=True),
            ToTensor(normalize={'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]})

        ], p=p)

    ],
    'default': [
        Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            Resize(width=224, height=224, always_apply=True),
            ToTensor(normalize={'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]})
        ], p=p),

        Compose([
            Resize(width=224, height=224, always_apply=True),
            ToTensor(normalize={'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]})

        ], p=p)
    ]

}
