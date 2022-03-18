import os
import sys
import shutil
import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("torch")
root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..")

OPTIMIZER = {
    "adam":torch.optim.Adam,
    "rmsprop":torch.optim.RMSprop,
    "sgd":torch.optim.SGD
}