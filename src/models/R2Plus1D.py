import math
import torch 
import torch.nn as nn
from typing import Tuple, List, Union
from torch.nn.modules.utils import _triple
from pytorch_model_summary import summary
import torch.nn.functional as F

# Block Component
# ConvBlock for Image
class ConvBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int = 1, dilation : int = 1, padding : int = 1, bias : bool = False, alpha : float = 0.01):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias = bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(alpha)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        return x

# ConvBlock for Image sequence
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

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        return x

# Residule Block for Conv3d (Image sequence)
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

    def forward(self, x:torch.Tensor)->torch.Tensor:
        res = self.conv1(x)
        res = self.bn2(self.conv2(res))
        if self.downsample:
            xi =  self.downsample_conv(x)
        else:
            xi = x
        return self.relu2(xi + res)

# Spatio Temporal Convolution encoder
# consist of temporal conv and spatio conv with conv3dblock
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

    def forward(self, x:torch.Tensor)->torch.Tensor:
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

    def forward(self, x:torch.Tensor)->torch.Tensor:
        res = self.conv1(x)
        res = self.conv2(res)
        if self.downsample:
            x = self.downsample_conv(x)

        return self.relu(x+res)

# Spatio Temporal Residue Layer for R2Plus1D model
class SpatioTemporalResLayer(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : Union[Tuple[int,int,int], int] = (3,1,1), downsample : bool = False, dilation : int = 1, alpha :float = 0.01, layer_size : int = 4):
        super(SpatioTemporalResLayer, self).__init__()
        self.block1 = SpatioTemporalResBlock(in_channels, out_channels, kernel_size, downsample = downsample, dilation = dilation, alpha = alpha)
        self.blocks = nn.ModuleList([])

        for _ in range(layer_size - 1):        
            self.blocks.append(
                SpatioTemporalResBlock(out_channels, out_channels, kernel_size, downsample = False, dilation = dilation, alpha = alpha)
            )
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x

# R2Plus1D Network : Spatio Temporal Conv + Spatio Temporal Resnet Layer
class R2Plus1DNet(nn.Module):
    def __init__(self, layer_sizes : List[int] = [4,4,4,4], alpha : float = 0.01):
        super(R2Plus1DNet, self).__init__()
        self.conv1 = SpatioTemporalConv(3, 64, kernel_size = (1,7,7), stride = (1,2,2), padding = (0,3,3), dilation = 1, is_first = True, alpha = alpha)
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, dilation = 1, alpha = alpha, layer_size = layer_sizes[0])
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, dilation = 1, alpha = alpha, layer_size = layer_sizes[1], downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, dilation = 1, alpha = alpha, layer_size = layer_sizes[2], downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, dilation = 1, alpha = alpha, layer_size = layer_sizes[3], downsample=True)
        self.pool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(batch_size, -1)
        return x

class R2Plus1DClassifier(nn.Module):
    def __init__(
        self, 
        input_size : Tuple[int, int, int, int] = (3, 8, 112, 112),
        num_classes : int = 2, 
        layer_sizes : List[int] = [4,4,4,4], 
        pretrained : bool = False, 
        alpha : float = 0.01
        ):
        super(R2Plus1DClassifier, self).__init__()
        self.input_size = input_size
        self.res2plus1d = R2Plus1DNet(layer_sizes, alpha = alpha)

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

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.res2plus1d(x)
        x = self.linear(x)
        return x

    def summary(self)->None:
        input_size = (1, *self.input_size)
        sample = torch.zeros(input_size).to(next(self.parameters()).device)
        print(summary(self, sample, max_depth = None, show_parent_layers = False, show_input = True))

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
        self.device = next(self.localization.parameters()).device
        sample_shape  =  (1, *(self.input_shape))
        sample_inputs = torch.zeros(sample_shape).to(self.device)
        sample_outputs = self.forward_localization(sample_inputs)
        return sample_outputs.view(sample_inputs.size(0), self.seq_len, -1).size()
        
    def forward_localization(self, x:torch.Tensor)->torch.Tensor:
        for layer in self.localization:
            x = layer.forward(x)
        return x

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x_sampled = torch.zeros_like(x).to(x.device)
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
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
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

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.res2plus1d(x)
        x = self.linear(x)
        return x

    def summary(self)->None:
        input_size = (1, *self.input_size)
        sample = torch.zeros(input_size).to(next(self.linear.parameters()).device)
        print(summary(self, sample, max_depth = None, show_parent_layers = False, show_input = True))

if __name__ == "__main__":

    model = R2Plus1DClassifier(
        input_size = (3, 21, 224, 224),
        num_classes = 2, 
        layer_sizes = [1,2,2,1], 
        pretrained = False, 
        alpha = 0.01
    )

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = 'cpu'
    
    model.to(device)
    model.summary()

    model.cpu()
    del model

    model = R2P1DwithSTNClassifier(
        input_size = (3, 21, 224, 224),
        num_classes = 2, 
        layer_sizes = [1,2,2,1], 
        pretrained = False, 
        alpha = 0.01
    )

    model.to(device)
    model.summary()