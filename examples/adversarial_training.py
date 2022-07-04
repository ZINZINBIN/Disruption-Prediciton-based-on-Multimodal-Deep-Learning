import numpy as np
import torch
import torch.nn as nn
import os, random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm
from typing import Optional
import torch.nn.functional as F
from pytorch_model_summary import summary

if __name__ == "__main__":
    print("# Adversarial training example # ")