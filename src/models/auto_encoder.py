import numpy as np
import torch 
import torch.nn as nn
from typing import List, Tuple, Optional

# For Video Input

# Generator
class VideoGenerater(nn.Module):
    def __init__(self):
        super(self, VideoGenerater).__init__()

    def forward(self, inputs : torch.Tensor):
        return None

# Discriminator
class VideoDiscriminator(nn.Module):
    def __init__(self):
        super(self, VideoDiscriminator).__init__()

    def forward(self, inputs : torch.Tensor):
        return None

from src.models.resnet import Bottleneck2DPlus1D
from src.models.model import ResNet50

class Video2ImageEncoder(nn.Module):
    def __init__(
        self, 
        input_shape : Tuple[int,int,int,int] = (3, 8, 112, 112)
        ):
        super(Video2ImageEncoder, self).__init__()
        self.input_shape = input_shape
        self.seq_len = input_shape[1]
        self.height = input_shape[2]
        self.width = input_shape[3]

        in_channels = 3
        hiddens = 128
        out_channels = 64

        self.conv1 = nn.Conv2d(in_channels, hiddens, kernel_size = (11,11))
        self.bn1 = nn.BatchNorm2d(hiddens)
        self.conv2 = nn.Conv2d(hiddens, out_channels, kernel_size = (5,5))
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def get_output_size(self):
        input_shape = (4, *(self.input_shape))
        device = next(self.resnet.parameters()).device
        sample = torch.zeros(input_shape).to(device)
        sample_output = self.forward(sample)
        return sample_output.size()
        
    def forward(self, x:torch.Tensor):
        outputs = torch.zeros()
        
        return x

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dims : int, hidden_dims : int, kernel_size : List[int], bias = None):
        super(ConvLSTMCell, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(input_dims + hidden_dims, 4 * hidden_dims, kernel_size, padding = self.padding, bias = self.bias)

    def forward(self, inputs:torch.Tensor, h : torch.Tensor, c : torch.Tensor)->torch.Tensor:

        combined = torch.cat([inputs, h], dim = 1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dims, dim = 1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dims, height, width, device = self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dims, height, width, device = self.conv.weight.device)
        )

class ConvLSTM(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : List[int], bias = None):
        super(ConvLSTM, self).__init__()
        self.out_channels = out_channels
        self.conv_lstm_cell = ConvLSTMCell(in_channels, out_channels, kernel_size, bias)
    
    def forward(self, x : torch.Tensor):
        B, _, seq_len, h, w = x.size()

        outputs = torch.zeros(B,self.out_channels, seq_len, h, w, device = self.conv_lstm_cell.conv.weight.device)
        H = torch.zeros(B, self.out_channels, seq_len, h, w, device = self.conv_lstm_cell.conv.weight.device)
        C = torch.zeros(B, self.out_channels, seq_len, h, w, device = self.conv_lstm_cell.conv.weight.device)

        for timestep in range(seq_len):
            H, C = self.conv_lstm_cell(x[:,:,timestep, :, :], H, C)
            outputs[:,:,timestep] = H

        return outputs

class AutoEncoder(nn.Module):
    def __init__(
        self, 
        input_shape : Tuple[int,int,int,int] = (3, 8, 112, 112),
        alpha : int = 1,
        layers : List[int] = [3,4,6,3],
        block : Optional[Bottleneck2DPlus1D] = Bottleneck2DPlus1D,
        ):
        super(AutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.alpha = alpha
        self.layers = layers
        self.block = block
        
