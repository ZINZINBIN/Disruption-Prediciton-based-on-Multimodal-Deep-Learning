import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, Union, List

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma : Optional[float]= 2, size_average : bool = True, ignore_label = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, input : torch.Tensor, target:torch.Tensor)->torch.Tensor:
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1) # N, C, H, W -> N, C, H * W
            input = input.transpose(1,2) # N, H*W, C
            input = input.contiguous().view(-1, input.size(2)) # N * H * W, C
        target = target.view(-1,1)

        if input.squeeze(1).dim() == 1:
            logpt = torch.sigmoid(input)
            logpt = logpt.view(-1)
        else:
            logpt = F.log_softmax(input, dim = 1)
            logpt = logpt.gather(1, target)
            logpt = logpt.view(-1)
        
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = (-1) * (1 - pt) ** self.gamma * logpt

        if self.ignore_label is not None:
            loss = loss[target[:,0] != self.ignore_label]
        
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# This code is based on LDAM
# reference : https://github.com/kaidic/LDAM-DRW/blob/master/losses.py

# compute focal loss
def compute_focal_loss(inputs:torch.Tensor, gamma:float):
    p = torch.exp(-inputs)
    loss = (1-p) ** gamma * inputs
    return loss.mean()

# focal loss object
class FocalLossLDAM(nn.Module):
    def __init__(self, weight : Optional[torch.Tensor] = None, gamma : float = 0.1):
        super(FocalLossLDAM, self).__init__()
        assert gamma >= 0, "gamma should be positive"
        self.gamma = gamma
        self.weight = weight

    def forward(self, input : torch.Tensor, target : torch.Tensor)->torch.Tensor:
        return compute_focal_loss(F.cross_entropy(input, target, reduction = 'none', weight = self.weight), self.gamma)

# Label-Distribution-Aware Margin loss
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list : Optional[List], max_m : float = 0.5, weight : Optional[torch.Tensor] = None, s : int = 30):
        super(LDAMLoss, self).__init__()
        assert s > 0, "s should be positive"
        self.s = s
        self.weight = weight
        
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list

    def forward(self, x : torch.Tensor, target : torch.Tensor)->torch.Tensor:
        idx = torch.zeros_like(x, dtype = torch.uint8)
        idx.scatter_(1, target.data.view(-1,1), 1)

        idx_float = idx.type(torch.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], idx_float.transpose(0,1))
        batch_m = batch_m.view((-1,1))
        x_m = x - batch_m

        output = torch.where(idx, x_m, x)

        return F.cross_entropy(self.s * output, target, weight = self.weight)