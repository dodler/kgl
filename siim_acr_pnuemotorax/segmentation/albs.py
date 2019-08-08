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

SIZE = 1024

aug_resize=Resize(SIZE, SIZE)

aug_light=Compose([
    aug_resize,
    OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            ], p=0.4),
    CLAHE(p=0.4),
    RandomBrightnessContrast(p=0.4),
    OneOf([RandomSizedCrop(min_max_height=(int(SIZE * 0.95), SIZE), height=SIZE, width=SIZE, p=0.5),
           PadIfNeeded(min_height=SIZE, min_width=SIZE, p=0.5)], p=1),
    ShiftScaleRotate(rotate_limit=40,p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    GaussNoise(p=0.5),
])

aug_geom_color = Compose([
    # Resize(SIZE, SIZE),
    OneOf([RandomSizedCrop(min_max_height=(int(SIZE*0.9), SIZE), height=SIZE, width=SIZE, p=0.5),
           PadIfNeeded(min_height=SIZE, min_width=SIZE, p=0.5)], p=1),
    # VerticalFlip(p=0.2),
    # RandomRotate90(p=0.5),
    # OneOf([
    #     ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    #     GridDistortion(p=0.5),
    #     OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
    #     ], p=0.8),
    # CLAHE(p=0.8),
    # RandomBrightnessContrast(p=0.4),
    RandomGamma(p=0.2),
])

aug_geometric = Compose([
    OneOf([RandomSizedCrop(min_max_height=(int(SIZE*0.9), SIZE), height=SIZE, width=SIZE, p=0.5),
           PadIfNeeded(min_height=SIZE, min_width=SIZE, p=0.5)], p=1),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    OneOf([
        ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(p=0.5),
        OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)
        ], p=0.8)])

