from albumentations import Compose, ShiftScaleRotate
from albumentations.pytorch import ToTensor

transform_train = Compose([
    ShiftScaleRotate(),
    ToTensor()
])
transform_test= Compose([
    ToTensor()
])