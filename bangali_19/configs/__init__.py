import importlib

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_config(name):
    pack = importlib.import_module(name, package='configs.{}'.format(name))
    return pack.config


if __name__ == '__main__':
    print(get_config('v0'))
