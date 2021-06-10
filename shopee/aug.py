import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2


def get_aug(atype, size):
    print('using aug type', atype, 'size', size)
    if atype == 'base':
        return alb.Compose([
            alb.Resize(size, size, p=1),
            alb.Transpose(p=0.5),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.ShiftScaleRotate(p=0.5),
            alb.Normalize(p=1),
            ToTensorV2(p=1),
        ])
    else:
        raise Exception('not supported')
