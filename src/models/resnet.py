import torch
import torch.nn as nn
import torch.nn.init as nn_init
import math
import numpy as np
from pytorch_model_summary import summary
from typing import Optional

def conv1x3x3(in_planes, out_planes, stride = 1):
    return nn.Conv3d(in_planes, out_planes, kernel_size = (1,3,3), stride = (1, stride,  stride), padding = (0,1,1), bias = False)

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_planes : int, planes : int, stride : int = 1, downsample : Optional[nn.Module]= None):
        super(BasicBlock3D, self).__init__()
        #self.expansion = 1
        self.conv1 = conv1x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    def init_temporal(self, strategy):
        raise NotImplementedError


class Bottleneck3D(nn.Module):
    expansion = 4
    def __init__(self, in_planes : int, planes : int, stride:int=1, downsample: Optional[nn.Module]=None, bias:bool=False, head_conv:int=1):
        super(Bottleneck3D, self).__init__()

        if head_conv == 1:
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")

        self.conv2 = nn.Conv3d(planes, planes,
                               kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=bias)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, **kwargs):
        super(ResNet3D, self).__init__()
        in_channels = kwargs['in_channels']
        self.alpha = kwargs['alpha']
        self.slow = kwargs['slow']  # slow->1 else fast->0

        self.inplanes = (64 + 64//self.alpha) if self.slow else 64//self.alpha

        # layer 0 parameters
        out_channels = 64//(1 if self.slow else self.alpha)
        kernel_size = (1 if self.slow else 1, 7, 7)
        stride = (1, 2, 2)
        padding = (0, 3, 3)
        
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # layer 1 to layer 4
        self.layer1 = self._make_layer(block, 64//(1 if self.slow else self.alpha), layers[0],
                                       head_conv=1 if self.slow else 3)
        self.layer2 = self._make_layer(block, 128//(1 if self.slow else self.alpha), layers[1], stride=2,
                                       head_conv=1 if self.slow else 3)
        self.layer3 = self._make_layer(block, 256//(1 if self.slow else self.alpha), layers[2], stride=2,
                                       head_conv=3)
        self.layer4 = self._make_layer(block, 512//(1 if self.slow else self.alpha), layers[3], stride=2,
                                       head_conv=3)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn_init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn_init.constant_(m.weight, 1)

    def forward(self, x):
        raise NotImplementedError('use each pathway network\' forward function')

    def _make_layer(self, block : Bottleneck3D, planes : int, blocks:int = 3, stride:int=1, head_conv:int =1):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=(1, stride, stride),bias=False), 
                    nn.BatchNorm3d(planes * block.expansion)
                )
        else:
            downsample = None

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, head_conv=head_conv))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, head_conv=head_conv))

        self.inplanes += self.slow * block.expansion * planes // self.alpha

        return nn.Sequential(*layers)