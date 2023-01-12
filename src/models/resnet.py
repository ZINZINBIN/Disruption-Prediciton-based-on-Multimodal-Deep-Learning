import torch
import torch.nn as nn
import torch.nn.init as nn_init
import math
import numpy as np
from pytorch_model_summary import summary
from typing import Optional

torch.backends.cudnn.benchmark = True

class SubBatchNorm3d(nn.Module):
    def __init__(self, num_splits, **args):
        super(SubBatchNorm3d, self).__init__()
        self.num_splits = num_splits
        self.num_features = args["num_features"]

        if args.get("affine", True):
            self.affine = True
            args["affine"] = False
            self.weight = torch.nn.Parameter(torch.ones(self.num_features))
            self.bias = torch.nn.Parameter(torch.zeros(self.num_features))
        
        else:
            self.affine = False
        
        self.bn = nn.BatchNorm3d(**args)
        args["num_features"] = self.num_features * self.num_splits
        self.split_bn = nn.BatchNorm3d(**args)

    def forward(self, x:torch.Tensor)->None:
        if self.training:
            n,c,t,h,w = x.shape
            x = x.view(n//self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n,c,t,h,w)
        else:
            x = self.bn(x)
        
        if self.affine:
            x = x * self.weight.view((-1,1,1,1))
            x = x + self.bias.view((-1,1,1,1))
        
        return x

    def _get_aggregated_mean_std(self, means : torch.Tensor, stds : torch.Tensor, n : int):
        mean = means.view(n, -1).sum(0) / n
        std = (
            stds.view(n,-1).sum(0) / n + ((means.view(n,-1) - mean) ** 2).view(n,-1).sum(0)/n
        )
        return mean.detach(), std.detach()
    
    def aggregate_stats(self):
        if self.split_bn.track_running_stats:
            (
                self.bn.running_mean.data,
                self.bn.running_var.data,
            ) = self._get_aggregated_mean_std(
                self.split_bn.running_mean,
                self.split_bn.running_var,
                self.num_splits
            )

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return SwishEfficient.apply(x)
    
class SwishEfficient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))

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
    def __init__(self, in_planes : int, planes : int, stride:int=1, downsample: Optional[nn.Module]=None, bias:bool=False, head_conv:int=1, base_bn_splits : Optional[int] = None, index : int = 0):
        super(Bottleneck3D, self).__init__()
        self.index = index

        if head_conv == 1:
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes) if base_bn_splits is None else SubBatchNorm3d(num_splits = base_bn_splits, num_features = planes, affine = True)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes) if base_bn_splits is None else SubBatchNorm3d(num_splits = base_bn_splits, num_features = planes, affine = True)
        else:
            raise ValueError("Unsupported head_conv!")

        self.conv2 = nn.Conv3d(planes, planes,
                               kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=bias)
        self.bn2 = nn.BatchNorm3d(planes) if base_bn_splits is None else SubBatchNorm3d(num_splits = base_bn_splits, num_features = planes, affine = True)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4) if base_bn_splits is None else SubBatchNorm3d(num_splits = base_bn_splits, num_features = planes * 4, affine = True)
        self.swish = Swish()
        self.relu = nn.ReLU(inplace=True)

        if self.index % 2 == 0:
            width = self.round_width(planes)
            self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
            self.fc1 = nn.Conv3d(planes, width, kernel_size = 1, stride = 1)
            self.fc2 = nn.Conv3d(width, planes, kernel_size = 1, stride = 1)
            self.sigmoid = nn.Sigmoid()

        self.downsample = downsample
        self.stride = stride

    def round_width(self, width : int, multiplier = 0.0625, min_width = 8, divisor = 8):

        if not multiplier:
            return width
        
        width *= multiplier
        min_width = min_width or divisor

        width_out = max(
            min_width, int(width + divisor / 2) // divisor * divisor
        )

        if width_out < 0.9 * width:
            width_out += divisor
        
        return int(width_out)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.index % 2 == 0:
            se_w = self.global_pool(out)
            se_w = self.fc1(se_w)
            se_w = self.relu(se_w)
            se_w = self.fc2(se_w)
            se_w = self.sigmoid(se_w)
            out = out * se_w

        out = self.swish(out)
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
        self.base_bn_splits = kwargs["base_bn_splits"]

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
                                       head_conv=1 if self.slow else 3, base_bn_splits=self.base_bn_splits)
        self.layer2 = self._make_layer(block, 128//(1 if self.slow else self.alpha), layers[1], stride=2,
                                       head_conv=1 if self.slow else 3, base_bn_splits=self.base_bn_splits)
        self.layer3 = self._make_layer(block, 256//(1 if self.slow else self.alpha), layers[2], stride=2,
                                       head_conv=3, base_bn_splits=self.base_bn_splits)
        self.layer4 = self._make_layer(block, 512//(1 if self.slow else self.alpha), layers[3], stride=2,
                                       head_conv=3, base_bn_splits=self.base_bn_splits)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn_init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d) and m.weight is not None:
                nn_init.constant_(m.weight, 1)

    def forward(self, x):
        raise NotImplementedError('use each pathway network\' forward function')

    def _make_layer(self, block : Bottleneck3D, planes : int, blocks:int = 3, stride:int=1, head_conv:int =1, base_bn_splits : Optional[int] = None):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=(1, stride, stride), bias=False), 
                    nn.BatchNorm3d(planes * block.expansion)
                )
        else:
            downsample = None

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, head_conv=head_conv, base_bn_splits = base_bn_splits))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, head_conv=head_conv, base_bn_splits = base_bn_splits))

        self.inplanes += self.slow * block.expansion * planes // self.alpha

        return nn.Sequential(*layers)

    def update_bn_splits_long_cycle(self, long_cycle_bn_scale):
        for m in self.modules():
            if isinstance(m, SubBatchNorm3d):
                m.num_splits = self.base_bn_splits * long_cycle_bn_scale
                m.split_bn = nn.BatchNorm3d(num_features = m.num_features * m.num_splits, affine = False).to(m.weight.device)
        
        return self.base_bn_splits * long_cycle_bn_scale


class Bottleneck2DPlus1D(nn.Module):
    expansion = 4
    def __init__(self, in_planes : int, planes : int, stride:int=1, downsample: Optional[nn.Module]=None, bias:bool=False, head_conv:int=1, index : int = 0):
        super(Bottleneck2DPlus1D, self).__init__()
        self.index = index

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
        self.swish = Swish()
        self.relu = nn.ReLU(inplace=True)

        if self.index % 2 == 0:
            width = self.round_width(planes)
            self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
            self.fc1 = nn.Conv3d(planes, width, kernel_size = 1, stride = 1)
            self.fc2 = nn.Conv3d(width, planes, kernel_size = 1, stride = 1)
            self.sigmoid = nn.Sigmoid()

        self.downsample = downsample
        self.stride = stride

    def round_width(self, width : int, multiplier = 0.0625, min_width = 8, divisor = 8):

        if not multiplier:
            return width
        
        width *= multiplier
        min_width = min_width or divisor

        width_out = max(
            min_width, int(width + divisor / 2) // divisor * divisor
        )

        if width_out < 0.9 * width:
            width_out += divisor
        
        return int(width_out)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.index % 2 == 0:
            se_w = self.global_pool(out)
            se_w = self.fc1(se_w)
            se_w = self.relu(se_w)
            se_w = self.fc2(se_w)
            se_w = self.sigmoid(se_w)
            out = out * se_w

        out = self.swish(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet2DPlus1D(nn.Module):
    def __init__(self, block, layers, **kwargs):
        super(ResNet2DPlus1D, self).__init__()
        in_channels = kwargs['in_channels']
        self.alpha = kwargs['alpha']

        self.inplanes = 64//self.alpha

        # layer 0 parameters
        out_channels = 64 // self.alpha
        kernel_size = (1, 7, 7)
        stride = (1, 2, 2)
        padding = (0, 3, 3)
        
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # layer 1 to layer 4
        self.layer1 = self._make_layer(block, 64 //self.alpha, layers[0], head_conv=1)
        self.layer2 = self._make_layer(block, 128 // self.alpha, layers[1], stride=2,head_conv=1)
        self.layer3 = self._make_layer(block, 256 // self.alpha, layers[2], stride=2,head_conv=3)
        self.layer4 = self._make_layer(block, 512 // self.alpha, layers[3], stride=2,head_conv=3)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn_init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d) and m.weight is not None:
                nn_init.constant_(m.weight, 1)

    def forward(self, x):
        raise NotImplementedError('use each pathway network\' forward function')

    def _make_layer(self, block : Bottleneck3D, planes : int, blocks:int = 3, stride:int=1, head_conv:int =1):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=(1, stride, stride), bias=False), 
                    nn.BatchNorm3d(planes * block.expansion)
                )
        else:
            downsample = None

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, head_conv=head_conv))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, head_conv=head_conv))

        return nn.Sequential(*layers)