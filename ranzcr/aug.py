import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2


def get_aug(cfg):
    print('getting aug', cfg['aug']['name'])
    print('aug size', cfg['aug']['size'])
    size = cfg['aug']['size']
    name = cfg['aug']['name']

    if name == 'v0':
        return alb.Compose([
            alb.RandomResizedCrop(size, size, p=1.0),
            alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
    elif name == 'v1':
        return alb.Compose([
            alb.RandomResizedCrop(size, size, p=1.0),
            alb.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), contrast_limit=(-0.4, 0.4), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=0.4, sat_shift_limit=0.4, val_shift_limit=0.5, p=0.3),
            alb.ShiftScaleRotate(p=0.3),
            alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
    else:
        raise Exception('{} not supported'.format(name))

    return aug
