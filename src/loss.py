# This code is based on LDAM
# reference : https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List

# Focal Loss : https://arxiv.org/abs/1708.02002 
class FocalLoss(nn.Module):
    def __init__(self, weight : Optional[torch.Tensor] = None, gamma : float = 2.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0, "gamma should be positive"
        self.loss_type = "Focal"
        self.gamma = gamma
        self.weight = weight
    
    def update_weight(self, weight : Optional[torch.Tensor] = None):
        self.weight = weight

    def compute_focal_loss(self, inputs:torch.Tensor, gamma:float, alpha : torch.Tensor):
        p = torch.exp(-inputs)
        loss = alpha * (1-p) ** gamma * inputs
        return loss.sum()

    def forward(self, input : torch.Tensor, target : torch.Tensor)->torch.Tensor:
        weight = self.weight.to(input.device)
        alpha = weight.gather(0, target.data.view(-1))
        alpha = Variable(alpha)
        return self.compute_focal_loss(F.cross_entropy(input, target, reduction = 'none', weight = None), self.gamma, alpha)

# Label-Distribution-Aware Margin loss
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list : Optional[List], max_m : float = 0.5, weight : Optional[torch.Tensor] = None, s : int = 30):
        super(LDAMLoss, self).__init__()
        assert s > 0, "s should be positive"
        self.model_type = "LDAM"
        self.s = s
        self.max_m = max_m
        self.weight = weight

        if cls_num_list:
            self.update_m_list(cls_num_list)
        
    def update_weight(self, weight : Optional[torch.Tensor] = None):
        self.weight = weight

    def update_m_list(self, cls_num_list : List):
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (self.max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
    
    def forward(self, x : torch.Tensor, target : torch.Tensor)->torch.Tensor:
        idx = torch.zeros_like(x, dtype = torch.uint8).to(x.device)
        idx.scatter_(1, target.data.view(-1,1), 1)

        idx_float = idx.type(torch.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], idx_float.transpose(0,1))
        batch_m = batch_m.view((-1,1)).to(x.device)
        x_m = x - batch_m

        output = torch.where(idx, x_m, x)

        return F.cross_entropy(self.s * output, target, weight = self.weight)