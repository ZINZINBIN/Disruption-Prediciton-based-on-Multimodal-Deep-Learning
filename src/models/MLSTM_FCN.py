''' Multivariate LSTM-FCNs for Time series classification
    Attention-LSTM based multivariate time series classification model
    The Convolution block with squeeze-and-excitation block is used for enhancement
    Reference
    - paper : https://arxiv.org/pdf/1801.04503v2.pdf
    - code : https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py
    - papers-with-codes : https://paperswithcode.com/paper/multivariate-lstm-fcns-for-time-series
'''

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        
    def forward(self, x : torch.Tensor):
        pass

class MLSTM_FCN(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x : torch.Tensor):
        pass