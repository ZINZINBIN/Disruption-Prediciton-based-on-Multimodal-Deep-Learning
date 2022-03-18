import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class ConvBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int = 1, dilation : int = 1, padding : int = 0, bias : bool = False, alpha : float = 0.01):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias = bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int = 1, dilation : int = 1, padding : int = 0, bias : bool = False, alpha : float = 0.01):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding , dilation, bias = bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x
    
# Spatial transformer layer
class SpatialTransformer(nn.Module):
    def __init__(self, 
    input_shape : Tuple[int, int, int] = (1, 28, 28),
    conv_channels : List[int] = [1, 8, 16], 
    conv_kernels : List[int] = [8, 4],
    pool_strides : List[int] =  [2, 2],
    pool_kernels : List[int] = [2, 2],
    theta_dim : int = 64
    ):
        super(SpatialTransformer, self).__init__()
        assert len(conv_channels) == len(conv_kernels) + 1, "length error"
        assert len(conv_channels) == len(pool_strides) + 1, "length error"
        assert len(conv_channels) == len(pool_kernels) + 1, "length error"

        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels
        self.pool_strides = pool_strides
        self.pool_kernels = pool_kernels
        self.input_shape = input_shape
        self.theta_dim =  theta_dim
        self.localization = nn.ModuleList()
        self.device = None

        for idx in range(len(conv_channels)-1):
            self.localization.append(
                nn.Conv2d(conv_channels[idx], conv_channels[idx+1], kernel_size=conv_kernels[idx])
            )
            self.localization.append(
                nn.MaxPool2d(pool_kernels[idx], pool_strides[idx])
            )
            self.localization.append(
                nn.ReLU()
            )

        local_output_shape = self.get_localization_output_size()

        # theta : attention score from  sample
        self.fc_loc = nn.Sequential(
            nn.Linear(local_output_shape[1], theta_dim),
            nn.ReLU(),
            nn.Linear(theta_dim, 6)
        )

    def get_localization_output_size(self):
        if self.device is None:
            self.device = next(self.localization.parameters()).device
        sample_shape  =  (1, *(self.input_shape))
        sample_inputs = torch.zeros(sample_shape).to(self.device)
        sample_outputs = self.forward_localization(sample_inputs)
        return sample_outputs.view(sample_inputs.size(0), -1).size()
        
    def forward_localization(self, x):
        for layer in self.localization:
            x = layer.forward(x)
        return x

    def forward(self, x):
        xs = self.forward_localization(x)
        xs = xs.view(x.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x