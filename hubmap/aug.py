from albumentations import *
import cv2


def get_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=0.1),
        VerticalFlip(p=0.1),
        RandomRotate90(p=0.1),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.1,
                         border_mode=cv2.BORDER_REFLECT),
    ], p=p)

