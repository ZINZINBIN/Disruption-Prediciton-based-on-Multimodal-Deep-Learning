import numpy as np
from torch import unsqueeze
import torch 
import torch.nn as nn
from typing import Tuple, List
from src.models.layer import *
from src.models.transformer import *
from pytorch_model_summary import summary

class R2P1DwithSTN(nn.Module):
    def __init__(self, input_shape : Tuple[int, int, int, int] = (3, 8, 112, 112), layer_sizes : List[int] = [4,4,4,4], alpha : float = 0.01):
        super(R2P1DwithSTN, self).__init__()
        self.STN = SpatialTransformer3D(
            input_shape=input_shape,
            conv_channels = [3,16,32],
            conv_kernels = [8,4],
            conv_strides=[1,1],
            conv_paddings=[1,1],
            pool_strides=[2,2],
            pool_kernels=[2,2],
            alpha = alpha,
            theta_dim = 128
        )

        self.conv1 = SpatioTemporalConv(3, 64, kernel_size = (1,7,7), stride = (1,2,2), padding = (0,3,3), dilation = 1, is_first = True, alpha = alpha)
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, dilation = 1, alpha = alpha, layer_size = layer_sizes[0])
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, dilation = 1, alpha = alpha, layer_size = layer_sizes[1], downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, dilation = 1, alpha = alpha, layer_size = layer_sizes[2], downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, dilation = 1, alpha = alpha, layer_size = layer_sizes[3], downsample=True)

        self.pool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.STN(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(batch_size, -1)

        return x

class R2P1DwithSTNClassifier(nn.Module):
    def __init__(
        self, 
        input_size : Tuple[int, int, int, int] = (3, 8, 112, 112),
        num_classes : int = 2, 
        layer_sizes : List[int] = [4,4,4,4], 
        pretrained : bool = False, 
        alpha : float = 0.01
        ):
        super(R2P1DwithSTNClassifier, self).__init__()
        self.input_size = input_size
        self.res2plus1d = R2P1DwithSTN(input_size, layer_sizes, alpha = alpha)

        linear_dims = self.get_res2plus1d_output_size()[1]

        self.linear = nn.Sequential(
            nn.Linear(linear_dims, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(alpha),
            nn.Linear(128, num_classes)
        )

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()
        
    def get_res2plus1d_output_size(self):
        input_size = (1, *self.input_size)
        sample = torch.zeros(input_size)
        sample_output = self.res2plus1d(sample)
        return sample_output.size()

    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x)->torch.Tensor:
        x = self.res2plus1d(x)
        x = self.linear(x)
        return x

    def summary(self)->None:
        input_size = (1, *self.input_size)
        sample = torch.zeros(input_size)
        print(summary(self, sample, max_depth = None, show_parent_layers = True, show_input = True))
