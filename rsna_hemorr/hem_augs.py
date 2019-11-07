from albumentations import Compose, ShiftScaleRotate, PadIfNeeded, RandomCrop, Resize, RandomSizedCrop, CLAHE, \
    RandomRotate90, Flip, OneOf, MotionBlur, MedianBlur, Blur, CenterCrop, LongestMaxSize, HorizontalFlip, VerticalFlip, \
    Transpose
from albumentations.pytorch import ToTensor

transform_train = Compose([
    RandomRotate90(0.2),
    Flip(p=0.2),
    ShiftScaleRotate(),
    OneOf([
        MotionBlur(p=.2),
        MedianBlur(blur_limit=3, p=0.1),
        Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    ToTensor()
])

transform_test = Compose([
    ToTensor()
])

IMG_SIZE_RAW = 224
RAW_CROP_SIZE = 448
transform_train_raw = Compose([
    Resize(RAW_CROP_SIZE,RAW_CROP_SIZE),
    # CenterCrop(width=IMG_SIZE_RAW, height=IMG_SIZE_RAW),
    RandomRotate90(0.2),
    Flip(p=0.2),
    ShiftScaleRotate(),
    ToTensor()
])

transform_test_raw = Compose([
    Resize(RAW_CROP_SIZE, RAW_CROP_SIZE),
    # CenterCrop(width=IMG_SIZE_RAW, height=IMG_SIZE_RAW),
    ToTensor()
])


def get_raw_tta():
    transform_test_raw = Compose([
        Resize(RAW_CROP_SIZE, RAW_CROP_SIZE),

        ToTensor()
    ])

    transform_test_hf = Compose([
        Resize(RAW_CROP_SIZE, RAW_CROP_SIZE),
        HorizontalFlip(p=1, always_apply=True),
        ToTensor()
    ])

    transform_test_vf = Compose([
        Resize(RAW_CROP_SIZE, RAW_CROP_SIZE),

        VerticalFlip(p=1, always_apply=True),
        ToTensor()
    ])

    transform_test_tr = Compose([
        Resize(RAW_CROP_SIZE, RAW_CROP_SIZE),
        Transpose(),
        ToTensor()
    ])

    return [transform_test_raw, transform_test_hf, transform_test_vf, transform_test_tr]


def get_png_tta():
    transform_test = Compose([
        ToTensor()
    ])

    tfm_hf = Compose([
        HorizontalFlip(p=1, always_apply=True),
        ToTensor()
    ])

    tfm_vf = Compose([
        VerticalFlip(p=1, always_apply=True),
        ToTensor()
    ])

    tfm_tr = Compose([
        Transpose(),
        ToTensor()
    ])

    return [transform_test, tfm_hf, tfm_vf, tfm_tr]
