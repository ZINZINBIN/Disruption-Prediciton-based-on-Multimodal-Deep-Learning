# This code is based on LDAM
# reference : https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List
import warnings

warnings.filterwarnings(action="ignore")

# Focal Loss : https://arxiv.org/abs/1708.02002
class FocalLoss(nn.Module):
    def __init__(self, weight : Optional[torch.Tensor] = None, gamma : float = 2.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0, "gamma should be positive"
        self.model_type = "Focal"
        self.gamma = gamma
        self.weight = weight
    
    def update_weight(self, weight : Optional[torch.Tensor] = None):
        self.weight = weight

    def compute_focal_loss(self, inputs:torch.Tensor, gamma:float, alpha : torch.Tensor):
        p = torch.exp(-inputs)
        loss = alpha * (1-p) ** gamma * inputs
        return loss.sum()

    def forward(self, input : torch.Tensor, target : torch.Tensor):
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
    
    def forward(self, x : torch.Tensor, target : torch.Tensor):
        idx = torch.zeros_like(x, dtype = torch.uint8).to(x.device)
        idx.scatter_(1, target.data.view(-1,1), 1)

        idx_float = idx.type(torch.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], idx_float.transpose(0,1))
        batch_m = batch_m.view((-1,1)).to(x.device)
        x_m = x - batch_m

        output = torch.where(idx, x_m, x)

        return F.cross_entropy(self.s * output, target, weight = self.weight)

# CCA Loss
# reference : https://github.com/Michaelvll/DeepCCA/
class CCALoss(nn.Module):
    def __init__(self, output_dim : int, use_all_singular_values : bool):
        super(CCALoss, self).__init__()
        self.output_dim = output_dim
        self.use_all_singular_values = use_all_singular_values
        
        self.r1 = 1e-3
        self.r2 = 1e-3
        self.eps = 1e-9
    
    def forward(self, h1 : torch.Tensor, h2 : torch.Tensor):
        h1, h2 = h1.t(), h2.t()
        
        o1 = h1.size(0)
        o2 = h1.size(0)
        
        m = h1.size(1)
        
        h1_ = h1 - h1.mean(dim = 1).unsqueeze(1)
        h2_ = h2 - h2.mean(dim = 1).unsqueeze(1)
        
        sigma_h12 = 1.0 / (m-1) * torch.matmul(h1_, h2_.t())
        sigma_h11 = 1.0 / (m-1) * torch.matmul(h1_, h1_.t() + self.r1 * torch.eye(o1, device = h1.device))
        
        sigma_h22 = 1.0 / (m-1) * torch.matmul(h2_, h2_.t() + self.r2 * torch.eye(o2, device = h1.device))
        
        [D1,V1] = torch.symeig(sigma_h11, eigenvectors=True)
        [D2,V2] = torch.symeig(sigma_h22, eigenvectors=True)
        
        pos_idx1 = torch.gt(D1, self.eps).nonzero()[:,0]
        D1 = D1[pos_idx1]
        V1 = V1[pos_idx1]
        
        pos_idx2 = torch.gt(D2, self.eps).nonzero()[:,0]
        D2 = D2[pos_idx2]
        V2 = V2[pos_idx2]
        
        sigma_h11_root_inv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        sigma_h22_root_inv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(sigma_h11_root_inv, sigma_h12), sigma_h22_root_inv)
        
        if self.use_all_singular_values:
            corr = torch.trace(
                torch.sqrt(torch.matmul(Tval.t(), Tval))
            )
        else:
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (
                torch.eye(trace_TT.shape[0]*self.r1).to(h1.device)
            ))
            
            U,V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>self.eps, U, (torch.ones(U.shape).double()*self.eps).to(h1.device))
            U = U.topk(self.output_dim)[0]
            corr = torch.sum(torch.sqrt(U))
            
        return -corr