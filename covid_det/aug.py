import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2


def get_train_aug(name, size):
    return A.Compose([
        A.Resize(width=size, height=size),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.05),
        A.ShiftScaleRotate(p=0.2),
        A.Normalize(mean=0.5, std=0.5, always_apply=True),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        min_area=0,
        min_visibility=0,
        label_fields=['labels']
    ))


def get_valid_aug(name, size):
    return A.Compose([
        A.Resize(width=size, height=size),
        A.Normalize(mean=0.5, std=0.5, always_apply=True),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        min_area=0,
        min_visibility=0,
        label_fields=['labels']
    ))
