import albumentations as alb
import cv2


def get_aug(p=1.0):
    return alb.OneOf([
        alb.HueSaturationValue(10, 15, 10),
        alb.CLAHE(clip_limit=2),
        alb.RandomBrightnessContrast(),
    ], p=0.3)
