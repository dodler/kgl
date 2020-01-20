import importlib

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_augmentation(name):
    pack = importlib.import_module(name, package='augmentations.{}'.format(name))
    return pack.train_aug, pack.valid_aug


if __name__ == '__main__':
    print(get_augmentation('v0'))
