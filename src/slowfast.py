import numpy as np
from torch import unsqueeze
import torch 
import torch.nn as nn
from typing import Tuple, List
from src.layer import *
from src.transformer import *
from src.resnet import *
from pytorch_model_summary import summary

# using slow-fast model : https://github.com/mbiparva/slowfast-networks-pytorch
class SlowNet(ResNet3D):
    def __init__(self, blocks, layers, **kwargs):
        super(SlowNet, self).__init__(blocks, layers, **kwargs)
        self.init_params()

    def forward(self, x)->torch.Tensor:
        x, laterals = x
        x = self.layer0(x)

        #print("after layer 0 x.size : ", x.size())
        x = torch.cat([x, laterals[0]], dim = 1)
        x = self.layer1(x)

        #print("after layer 1 x.size : ", x.size())
        x = torch.cat([x, laterals[1]], dim = 1)
        x = self.layer2(x)

        x = torch.cat([x, laterals[2]], dim = 1)
        x = self.layer3(x)

        x = torch.cat([x, laterals[3]], dim = 1)
        x = self.layer4(x)

        x = F.adaptive_avg_pool3d(x, 1)
        x = x.view(-1, x.size(1))

        return x

def resnet50_s(block = Bottleneck3D, layers = [3,4,6,3], **kwargs):
    model = SlowNet(block, layers, **kwargs)
    return model

class FastNet(ResNet3D):
    def __init__(self, blocks, layers, **kwargs):
        super(FastNet, self).__init__(blocks, layers, **kwargs)
        alpha = kwargs["alpha"]
        kernel_size = (alpha+2,1,1)
        stride = (alpha,1,1)
        padding = (1,0,0)

        stride_maxpool = (alpha, 1, 1)
        kernel_maxpool = (alpha + 2, 1, 1)

        self.l_maxpool = nn.Conv3d(64//self.alpha, 64//self.alpha,
                                   kernel_size=kernel_maxpool, stride=stride_maxpool, bias=False, padding=padding)
        self.l_layer1 = nn.Conv3d(4*64//self.alpha, 4*64//self.alpha,
                                  kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.l_layer2 = nn.Conv3d(8*64//self.alpha, 8*64//self.alpha,
                                  kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.l_layer3 = nn.Conv3d(16*64//self.alpha, 16*64//self.alpha,
                                  kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.init_params()

    def forward(self, x : torch.Tensor)->torch.Tensor:
        laterals = []

        x = self.layer0(x)
        laterals.append(self.l_maxpool(x))

        x = self.layer1(x)
        laterals.append(self.l_layer1(x))

        x = self.layer2(x)
        laterals.append(self.l_layer2(x))

        x = self.layer3(x)
        laterals.append(self.l_layer3(x))

        x = self.layer4(x)

        x = F.adaptive_avg_pool3d(x, 1)
        x = x.view(-1, x.size(1))

        return x, laterals

def resnet50_f(block = Bottleneck3D, layers = [3,4,6,3], **kwargs):
    model = FastNet(block, layers, **kwargs)
    return model