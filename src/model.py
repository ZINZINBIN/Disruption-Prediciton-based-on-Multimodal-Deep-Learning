# ======================================================================
# Image - Tabular time series encoder model
# data structure : time series image data + plasma parameter(beta, kappa, ...)
# Image Encoder Model : ResNet, utae
# Tabular data Encoder Model : TabNet or Transformer model
# ======================================================================
import numpy as np
import torch 
import torch.nn as nn
from typing import Tuple, List
from src.layer import *

class SpatialTemporalConv(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int = 1, padding : int = 0, bias : bool = False):
        super(SpatialTemporalConv, self).__init__()

    def forward(self, x):
        pass
    
class Transformer(nn.Module):
    def __init__(self, timesteps : int):
        super(Transformer, self).__init__()

    def forward(self, inputs):
        return None

    def predict(self, inputs):
        return None

class  MNIST_Net(nn.Module):
    def __init__(
        self,
        input_shape : Tuple[int, int, int] = (28, 28, 1),
        conv_channels : List[int] = [1, 8, 16], 
        conv_kernels : List[int] = [8, 4],
        pool_strides : List[int] =  [2, 2],
        pool_kernels : List[int] = [2, 2],
        theta_dim : int = 64
        ):
        super(MNIST_Net, self).__init__()
        self.STN = SpatialTransformer(
            input_shape,
            conv_channels,
            conv_kernels,
            pool_strides,
            pool_kernels,
            theta_dim
        )
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x:torch.Tensor):
        #x = x.permute(0,2,3,1)
        batch = x.size(0)
        x = self.STN(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(batch, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)