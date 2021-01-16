from albumentations.pytorch.transforms import ToTensorV2
import albumentations as alb


def get_aug(atype, size):
    print('using aug', atype)
    if atype == '0':
        return alb.Compose([
            alb.Resize(size, size, p=1),
            alb.Transpose(p=0.5),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.ShiftScaleRotate(p=0.5),
            alb.Normalize(p=1),
            ToTensorV2(p=1),
        ])
    elif atype == '1':
        return alb.Compose([
            alb.RandomResizedCrop(size, size),
            alb.Transpose(p=0.5),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.ShiftScaleRotate(p=0.5),
            alb.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            alb.CoarseDropout(p=0.5),
            alb.Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ])
    elif atype == '2':
        return alb.Compose([
            alb.RandomResizedCrop(size, size, p=1, scale=(0.9, 1)),
            alb.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.2),
            alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.2),
            alb.Transpose(p=0.2),
            alb.HorizontalFlip(p=0.2),
            alb.VerticalFlip(p=0.2),
            alb.ShiftScaleRotate(p=0.2),
            alb.Normalize(p=1),
            ToTensorV2(p=1),
        ])
    elif atype == '3':
        return alb.Compose([
            alb.RandomResizedCrop(size, size, p=1),
            alb.HorizontalFlip(p=0.2),
            alb.Normalize(p=1),
            ToTensorV2(p=1),
        ])
    else:
        raise Exception('atype |{}| not supoprted'.format(atype))
