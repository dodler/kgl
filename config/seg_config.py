import json
from collections import namedtuple


class SegConfig:
    def __init__(self):
        pass


def from_json(path):
    with open(path) as f:
        config=json.load(f)

    nt = namedtuple("Config", config.keys())(*config.values())
    return nt
