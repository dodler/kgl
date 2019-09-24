import cv2
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    Resize, ShiftScaleRotate, GaussNoise)

SIZE = 256

aug_resize = Resize(SIZE, SIZE)
aug_crop = RandomSizedCrop(min_max_height=(SIZE, SIZE), height=SIZE, width=SIZE, p=1.0)

aug_light = Compose([
    RandomSizedCrop(min_max_height=(SIZE, SIZE), height=SIZE, width=SIZE, p=1.0),
    HorizontalFlip(p=0.2),
    ShiftScaleRotate(rotate_limit=40, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0),
    GaussNoise(p=0.4),
])

aug_geom_color = Compose([
    OneOf([RandomSizedCrop(min_max_height=(int(SIZE * 0.9), SIZE), height=SIZE, width=SIZE, p=0.5),
           PadIfNeeded(min_height=SIZE, min_width=SIZE, p=0.5)], p=1),
    RandomGamma(p=0.2),
])

aug_geometric = Compose([
    OneOf([RandomSizedCrop(min_max_height=(int(SIZE * 0.9), SIZE), height=SIZE, width=SIZE, p=0.5),
           PadIfNeeded(min_height=SIZE, min_width=SIZE, p=0.5)], p=1),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    OneOf([
        ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(p=0.5),
        OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)
    ], p=0.8)])


CLS_SIZE=256

aug_light_cls = Compose([
    Resize(256,1600),
    # HorizontalFlip(p=0.2),
    ShiftScaleRotate(rotate_limit=40, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0),
    # GaussNoise(p=0.4),
])
