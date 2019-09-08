import torch, catalyst

import os
from typing import List, Tuple, Callable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 42
from catalyst.utils import set_global_seed, prepare_cudnn

set_global_seed(SEED)
prepare_cudnn(deterministic=True)