import os
import sys
import numpy as np
import random
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps

try:
    import accimage
except ImportError:
    accimage = None

def background_removal(
    x : torch.Tensor, 
    seq_len : int = 8,
    height : int = 112, 
    width : int = 112,
    rank : int = 4, 
    some : bool = True, 
    compute_uv : bool = True):

    device = x.device
    x = x.view(-1,1,1,height, width).squeeze(1).squeeze(1).cpu()
    for idx in range(x.size(0)):
        U,S,Vh = torch.svd(x[idx, :, :], some = some, compute_uv = compute_uv)
        low_rank_diag = torch.diag(S[0:rank])
        rest_diag = torch.zeros((S.size(0) - rank, S.size(0) - rank))
        block = torch.block_diag(low_rank_diag, rest_diag)

        x[idx, :, :] -= U @ block @ Vh

    x = x.contiguous().unsqueeze(1).unsqueeze(1).view(-1,3,seq_len,height,width).to(device)
    return x