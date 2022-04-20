import numpy as np
import torch 
from typing import Optional

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)

# video mixup algorithm
def video_mixup_data(
    x : torch.Tensor, y:torch.Tensor, 
    device : str = "cpu",
    mode : Optional[str] = "spatial",
    alpha : float = 1.0):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    B = x.size(0)
    C = x.size(1)
    T = x.size(2)
    H = x.size(4)
    W = x.size(3)

    index = torch.randperm(B).to(device)

    wc = np.random.uniform(0,W)
    hc = np.random.uniform(0,H)
    tc = np.random.uniform(0,T)

    if mode == "spatial":
        w1 = int(wc - W * np.sqrt(lam) / 2)
        w2 = int(wc + W * np.sqrt(lam) / 2)

        h1 = int(hc - H * np.sqrt(lam) / 2)
        h2 = int(hc + H * np.sqrt(lam) / 2)

        t1 = 0
        t2 = T

    elif mode == "temporal":
        t1 = int(tc - T * np.sqrt(lam) / 2)
        t2 = int(tc + T * np.sqrt(lam) / 2)
        w1 = 0
        w2 = W
        h1 = 0
        h2 = H

    elif mode == "spatial-temporal":
        t1 = int(tc - T * np.sqrt(lam) / 2)
        t2 = int(tc + T * np.sqrt(lam) / 2)
        w1 = int(wc - W * np.sqrt(lam) / 2)
        w2 = int(wc + W * np.sqrt(lam) / 2)
        h1 = int(hc - H * np.sqrt(lam) / 2)
        h2 = int(hc + H * np.sqrt(lam) / 2)

    else:
        t1 = int(tc - T * np.sqrt(lam) / 2)
        t2 = int(tc + T * np.sqrt(lam) / 2)
        w1 = int(wc - W * np.sqrt(lam) / 2)
        w2 = int(wc + W * np.sqrt(lam) / 2)
        h1 = int(hc - H * np.sqrt(lam) / 2)
        h2 = int(hc + H * np.sqrt(lam) / 2)

    x_copy = x.clone().detach()[index]
    y_copy = y.clone().detach()[index]

    x[:,:,t1:t2,w1:w2,h1:h2] = x_copy[:,:,t1:t2,w1:w2,h1:h2]

    lam = (t2 - t1) * (h2 - h1) * (w2 - w1) / T / H / W

    return x, y, y_copy, lam