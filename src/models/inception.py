import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, cast
from src.models.NoiseLayer import NoiseLayer
from pytorch_model_summary import summary

class Conv1dSamePadding(nn.Conv1d):
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)

def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride, padding=padding // 2, dilation=dilation, groups=groups)

class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels : int, 
        out_channels : Union[List[int], int],
        residual : bool,
        stride : int = 1,
        bottleneck_channels : int = 32,
        kernel_size : int = 41
        ):
        super(InceptionBlock, self).__init__()
        self.use_bottleneck = bottleneck_channels > 0
        
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels, kernel_size = 1, bias = False)
        
        kernel_size_s = [kernel_size // (2**i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        
        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(channels[i], channels[i+1], kernel_size = kernel_size_s[i], stride = stride, bias = False)
            for i in range(len(kernel_size_s))
        ])
        
        self.norm = nn.BatchNorm1d(channels[-1])
        self.relu = nn.ReLU()
        
        self.use_residual = residual
        
        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])
    
    def forward(self, x : torch.Tensor):
        x_origin = x
        
        if self.use_bottleneck:
            x = self.bottleneck(x)
            
        x = self.conv_layers(x)
        x = self.norm(x)
        x = self.relu(x)
        
        if self.use_residual:
            x = x + self.residual(x_origin)
                
        return x        

class TSInception(nn.Module):
    def __init__(
        self, 
        n_blocks : int, 
        in_channels : int, 
        out_channels : Union[List[int], int],
        bottleneck_channels : Union[List[int], int],
        kernel_sizes : Union[List[int], int],
        use_residuals : Union[List[bool], bool, str] = "default",
        seq_len : int = 21,
        hidden_dim : int = 128,
        n_classes : int = 2
        ):
        super(TSInception, self).__init__()
        
        self.args = {
            "n_blocks" : n_blocks,
            "in_channels" : in_channels,
            "out_channels" : out_channels,
            "bottlenect_channels" : bottleneck_channels,
            "kernel_sizes" : kernel_sizes,
            "use_residuals" : use_residuals,
            "n_classes" : n_classes,
            "seq_len" : seq_len
        }
        
        channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels, n_blocks))
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels, n_blocks))
        
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, n_blocks))
        
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(n_blocks)]
        else:
            use_residuals = cast(List[bool], self._expand_to_blocks(
                cast(Union[bool, List[bool]], use_residuals), n_blocks
            ))
        
        self.blocks = nn.Sequential(*[
            InceptionBlock(
                in_channels=channels[i], out_channels=channels[i+1],
                residual = use_residuals[i], bottleneck_channels = bottleneck_channels[i],
                kernel_size = kernel_sizes[i]
            )
            for i in range(n_blocks)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=channels[-1], out_features = hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features = n_classes),
        )
        
        self.noise = NoiseLayer(mean = 0, std = 0.01)
    
    @staticmethod
    def _expand_to_blocks(value : Union[int, bool, List[int], List[bool]], num_blocks : int):
        
        if isinstance(value, list):
            assert len(value) == num_blocks, "length of input list must be equal to num block"
        else:
            value = [value] * num_blocks
        
        return value    
    
    def forward(self, x : torch.Tensor):
        
        if x.size()[1] != self.args['in_channels'] and x.size()[2] != self.args['seq_len']:
            x = x.permute(0,2,1)
        
        x = self.noise(x)
        x = self.blocks(x).mean(dim = -1)
        x = self.classifier(x)
        return x
    
    def summary(self):
        sample_x = torch.zeros((2, self.args['in_channels'], self.args['seq_len']))
        summary(self, sample_x, batch_size = 2, show_input = True, print_summary=True)
        