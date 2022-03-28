import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
from torch.nn.modules.utils import _triple

class ConvBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int = 1, dilation : int = 1, padding : int = 1, bias : bool = False, alpha : float = 0.01):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias = bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size = 3, stride = 1, dilation : int = 1, padding = 1, bias : bool = False, alpha : float = 0.01):
        super(Conv3dBlock, self).__init__()

        if type(stride) == tuple:
            strides = stride
        else:
            strides = (1, stride, stride)

        if type(kernel_size) == tuple:
            kernel_sizes = kernel_size
        else:
            kernel_sizes = (1, kernel_size, kernel_size)

        if type(padding) == tuple:
            paddings = padding
        else:
            paddings = (0, padding, padding)

        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size = kernel_sizes, 
            stride =  strides, 
            padding = paddings, 
            dilation = dilation, 
            bias = bias)

        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class Conv3dTransposeBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size = 3, stride = 1, dilation : int = 1, padding = 1, bias : bool = False, alpha : float = 0.01):
        super(Conv3dTransposeBlock, self).__init__()

class Conv3dResBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int =  3, stride : int =  1, dilation : int = 1, padding :int =  1, bias : bool = False, alpha : float = 0.01, downsample : bool = True):
        super(Conv3dResBlock, self).__init__()
        self.downsample = downsample

        pad = kernel_size // 2

        if self.downsample:
            self.stride = (1, stride, stride)
            self.padding = (0, padding, padding)
        else:
            self.stride = (1,1,1)
            self.padding = (0, pad, pad)

        self.kernel_size = (1, kernel_size, kernel_size)

        if self.downsample:
            self.downsample_conv = Conv3dBlock(
                in_channels,
                out_channels,
                kernel_size = self.kernel_size,
                stride = self.stride,
                padding = self.padding,
                dilation=dilation,
                alpha = alpha,
                bias = bias
            )
    
        self.conv1 = Conv3dBlock(
            in_channels,
            out_channels,
            kernel_size = self.kernel_size,
            padding = self.padding,
            stride=self.stride,
            dilation=dilation,
            alpha = alpha,
            bias = bias
        )

        self.conv2 = nn.Conv3d(out_channels, out_channels, self.kernel_size, (1,1,1), (0,pad,pad), dilation, bias = bias )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.LeakyReLU(alpha)

    def forward(self, x:torch.Tensor):
        res = self.conv1(x)
        res = self.bn2(self.conv2(res))
        if self.downsample:
            xi =  self.downsample_conv(x)
        else:
            xi = x
        return self.relu2(xi + res)

class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size = (3,1,1), stride = (1,1,1), dilation : int = 1, padding = (1,1,1), bias : bool = False, alpha : float = 0.01, is_first : bool = False):
        super(SpatioTemporalConv, self).__init__()

        if type(kernel_size) == int:
            kernel_size = _triple(kernel_size)
        if type(stride) ==  int:
            stride = _triple(stride)
        if type(padding) == int:
            padding =  _triple(padding)

        if is_first:
            # spatio conv
            spatio_kernel_size = kernel_size
            spatio_stride = (1, stride[1], stride[2])
            spatio_padding = padding

            # temporal conv
            temporal_kernel_size = (3,1,1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (1,0,0)

            middle_channels = 45

            self.spatio_conv = Conv3dBlock(in_channels, middle_channels, spatio_kernel_size, spatio_stride, dilation, spatio_padding, False, alpha)
            self.temporal_conv = Conv3dBlock(middle_channels, out_channels, temporal_kernel_size, temporal_stride,  dilation, temporal_padding, False, alpha)
        else:
            spatio_kernel_size = (1, kernel_size[1], kernel_size[2])
            spatio_stride = (1, stride[1], stride[2])
            spatio_padding = (0, padding[1], padding[2])

            temporal_kernel_size = (kernel_size[0],1,1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (padding[0], 0, 0)

            middle_channels = int(
                math.floor(
                    (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / \
                        (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)
                )
            )
            self.spatio_conv = Conv3dBlock(in_channels, middle_channels, spatio_kernel_size, spatio_stride, dilation, spatio_padding, bias, alpha)
            self.temporal_conv = Conv3dBlock(middle_channels, out_channels, temporal_kernel_size, temporal_stride, dilation, temporal_padding, bias, alpha)

    def forward(self, x):
        x = self.spatio_conv(x)
        x = self.temporal_conv(x)
        return x

class SpatioTemporalResBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : Union[Tuple[int,int,int], int] = (3,1,1), downsample : bool = False, dilation : int = 1, alpha :float = 0.01):
        super(SpatioTemporalResBlock, self).__init__()
        self.downsample = downsample

        padding = kernel_size // 2

        if self.downsample:
            self.downsample_conv = SpatioTemporalConv(in_channels, out_channels, kernel_size = 1, stride = 2, dilation = dilation, padding = 0)
            
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, stride =  (2,2,2), dilation=dilation, padding = padding)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, stride = (1,1,1), dilation = dilation, padding =padding)
        
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, stride = (1,1,1), padding = padding, dilation = dilation)
        self.relu = nn.LeakyReLU(alpha)

    def forward(self, x):

        res = self.conv1(x)
        res = self.conv2(res)

        if self.downsample:
            x = self.downsample_conv(x)

        return self.relu(x+res)
        
class SpatioTemporalResLayer(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : Union[Tuple[int,int,int], int] = (3,1,1), downsample : bool = False, dilation : int = 1, alpha :float = 0.01, layer_size : int = 4):
        super(SpatioTemporalResLayer, self).__init__()
        self.block1 = SpatioTemporalResBlock(in_channels, out_channels, kernel_size, downsample = downsample, dilation = dilation, alpha = alpha)
        self.blocks = nn.ModuleList([])

        for _ in range(layer_size - 1):        
            self.blocks.append(
                SpatioTemporalResBlock(out_channels, out_channels, kernel_size, downsample = False, dilation = dilation, alpha = alpha)
            )
    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
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

class SpatialTransformer3D(nn.Module):
    def __init__(self, 
    input_shape : Tuple[int, int, int, int] = (3, 8, 112, 112),
    conv_channels : List[int] = [3, 16, 32], 
    conv_kernels : List[int] = [8, 4],
    conv_strides : List[int] = [1, 1],
    conv_paddings : List[int] = [1, 1],
    pool_strides : List[int] =  [2, 2],
    pool_kernels : List[int] = [2, 2],
    alpha : float = 0.01,
    theta_dim : int = 64
    ):

        super(SpatialTransformer3D, self).__init__()
        assert len(conv_channels) == len(conv_kernels) + 1, "length error"
        assert len(conv_channels) == len(pool_strides) + 1, "length error"
        assert len(conv_channels) == len(pool_kernels) + 1, "length error"

        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels
        self.conv_paddings = conv_paddings
        self.conv_strides = conv_strides
        self.pool_strides = pool_strides
        self.pool_kernels = pool_kernels
        self.input_shape = input_shape
        self.theta_dim =  theta_dim
        self.localization = nn.ModuleList()
        self.alpha = alpha
        self.device = None

        self.seq_len = input_shape[1]

        for idx in range(len(conv_channels)-1):

            in_channels = conv_channels[idx]
            out_channels = conv_channels[idx+1]
            kernel_size = (1, conv_kernels[idx], conv_kernels[idx])
            stride = (1, conv_strides[idx], conv_strides[idx])
            padding = (0, conv_paddings[idx],  conv_paddings[idx])

            pool_kernel = (1, pool_kernels[idx], pool_kernels[idx])
            pool_stride = (1, pool_strides[idx],  pool_strides[idx])


            self.localization.append(
                nn.Conv3d(in_channels, out_channels, kernel_size= kernel_size, stride = stride, padding = padding, dilation = 1, padding_mode = 'zeros')
            )
            self.localization.append(
                nn.MaxPool3d(pool_kernel, pool_stride)
            )
            self.localization.append(
                nn.LeakyReLU(alpha)
            )

        local_output_shape = self.get_localization_output_size()

        # theta : attention score from  sample
        self.fc_loc = nn.Sequential(
            nn.Linear(local_output_shape[-1], theta_dim),
            nn.ReLU(),
            nn.Linear(theta_dim, 6)
        )

    def get_localization_output_size(self):
        if self.device is None:
            self.device = next(self.localization.parameters()).device
        sample_shape  =  (1, *(self.input_shape))
        sample_inputs = torch.zeros(sample_shape).to(self.device)
        sample_outputs = self.forward_localization(sample_inputs)
        return sample_outputs.view(sample_inputs.size(0), self.seq_len, -1).size()
        
    def forward_localization(self, x:torch.Tensor)->torch.Tensor:
        for layer in self.localization:
            x = layer.forward(x)
        return x

    def forward(self, x:torch.Tensor):
        x_sampled = torch.zeros_like(x)
        xs = self.forward_localization(x)
        xs = xs.view(x.size(0), self.seq_len, -1)
        theta = self.fc_loc(xs)

        for idx in range(self.seq_len):
            x_spatio = x[:,:,idx,:,:].squeeze(2)
            x_theta = theta[:,idx,:].view(-1,2,3)
            grid = F.affine_grid(x_theta, x_spatio.size())
            grid = F.grid_sample(x_spatio, grid)
            # x_sampled[:,:,idx,:,:] = grid.unsqueeze(2)
            x_sampled[:,:,idx,:,:] = grid
            
        return x_sampled


'''SlowFast model From Facebook AI  Research Team
- different seq_distance to extract different features from slow and fast model
- codes : github()
'''
# From SlowFast model
class SubBatchNorm3D(nn.Module):
    def __init__(self, num_split, **kwargs):
        super(SubBatchNorm3D, self).__init__()
        self.num_split = num_split
        self.num_features = kwargs['num_features']

        if kwargs.get("affine", True):
            self.affine = True
            kwargs['affine'] = False
            self.weight = nn.Parameter(torch.ones(self.num_features))
            self.bias = nn.Parameter(torch.zeros(self.num_features))
        
        else:
            self.affine = False

        self.bn = nn.BatchNorm3d(**kwargs)
        kwargs['num_features'] = self.num_features * self.num_split
        self.split_bn = nn.BatchNorm3d(**kwargs)

    def get_aggregated_mean_std(self, means, stds, n):
        mean = means.view(n, -1).sum(0) / n
        std = (
            stds.view(n,-1).sum(0) / n + ((means.view(n,-1) - mean)**2).view(n, -1).sum(0) /  n
        )

        return mean.detach(), std.detach()

    def aggregate_stats(self):
        if self.split_bn.track_running_stats:
            ( self.bn.running_mean.data, self.bn.running_var.data, ) = self.get_aggregated_mean_std(
                self.split_bn.running_mean,
                self.split_bn.running_var,
                self.num_split
            )
    def forward(self, x):
        if self.training:
            n,c,t,h,w = x.shape
            x  = x.view(n // self.num_split, c * self.num_split,  t, h, w)
            x = self.split_bn(x)
            x = x.view(n,c,t,h,w)

        else:
            x = self.bn(x)
        
        if self.affine:
            x = x * self.weight.view((-1,1,1,1))
            x = x + self.bias.view((-1,1,1,1))
        return x

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

class Bottleneck(nn.Module):
    def __init__(
        self, 
        in_planes : int, 
        planes : List[int], 
        kernel_size : int = 3,
        stride :int = 1, 
        dilation : int = 1,
        padding : int = 1,
        bias : bool = False,
        alpha : float = 0.01,
        downsample = None, 
        index = 0, 
        base_bn_splits = 8
        ):
        super(Bottleneck, self).__init__()
        self.index = index
        self.base_bn_splits = base_bn_splits
        self.conv1 = Conv3dBlock(in_planes, planes[0], kernel_size, stride, dilation, padding, bias, alpha)
        self.bn1 = SubBatchNorm3D(num_splits = base_bn_splits, num_features = planes[0], affine = True)
