import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from typing import List, Optional, Union, Tuple
from pytorch_model_summary import summary
from src.models.NoiseLayer import NoiseLayer

class CnnLSTM(nn.Module):
    def __init__(
        self,
        seq_len : int = 21, 
        n_features : int = 10,
        conv_dim : int = 32, 
        conv_kernel : int = 3,
        conv_stride : int = 1, 
        conv_padding : int = 1,
        lstm_dim : int = 64, 
        n_layers : int = 1,
        bidirectional : bool = True,
        n_classes : int = 2, 
        ):
    
        super(CnnLSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.lstm_dim = lstm_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_classes = n_classes
        
        self.noise = NoiseLayer(mean = 0, std = 1e-2)

        # spatio-conv encoder : analyze spatio-effect between variables
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = n_features, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.Conv1d(in_channels = conv_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim), 
            nn.ReLU(),
        )

        lstm_input_dim = self.compute_conv1d_output_dim(self.compute_conv1d_output_dim(seq_len, conv_kernel, conv_stride, conv_padding, 1), conv_kernel, conv_stride, conv_padding, 1)

        # temporl - lstm
        self.lstm = nn.LSTM(lstm_input_dim, lstm_dim, bidirectional = bidirectional, batch_first = False, num_layers = n_layers)
        
        if bidirectional:
            self.w_s1 = nn.Linear(lstm_dim * 2, lstm_dim)
            linear_input_dims = lstm_dim * 2
        else:
            self.w_s1 = nn.Linear(lstm_dim, lstm_dim)
            linear_input_dims = lstm_dim
            
        self.w_s2 = nn.Linear(lstm_dim, lstm_dim)

        self.classifier = nn.Sequential(
            nn.Linear(linear_input_dims, linear_input_dims//2),
            nn.BatchNorm1d(linear_input_dims//2),
            nn.ReLU(),
            nn.Linear(linear_input_dims//2, n_classes)
        )
    
    def compute_conv1d_output_dim(self, input_dim : int, kernel_size : int = 3, stride : int = 1, padding : int = 1, dilation : int = 1):
        return int((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def attention(self, lstm_output : torch.Tensor):
        attn_weight_matrix = self.w_s2(torch.tanh(self.w_s1(lstm_output)))
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim = 2)
        return attn_weight_matrix
    
    def encode(self, x:torch.Tensor):
        with torch.no_grad():
            x = self.noise(x)
            x_conv = self.conv(x.permute(0,2,1))
            h_0 = Variable(torch.zeros(2 * self.n_layers if self.bidirectional else self.n_layers, x.size()[0], self.lstm_dim)).to(x.device)
            c_0 = Variable(torch.zeros(2 * self.n_layers if self.bidirectional else self.n_layers, x.size()[0], self.lstm_dim)).to(x.device)

            lstm_output, (h_n,c_n) = self.lstm(x_conv.permute(1,0,2), (h_0, c_0))
            lstm_output = lstm_output.permute(1,0,2)
            att = self.attention(lstm_output)
            hidden = torch.bmm(att.permute(0,2,1), lstm_output).mean(dim = 1)
            hidden = hidden.view(hidden.size()[0], -1)
            
        return hidden

    def forward(self, x : torch.Tensor):
        # x : (batch, seq_len, col_dim)
        x = self.noise(x)
        x_conv = self.conv(x.permute(0,2,1))
        h_0 = Variable(torch.zeros(2 * self.n_layers if self.bidirectional else self.n_layers, x.size()[0], self.lstm_dim)).to(x.device)
        c_0 = Variable(torch.zeros(2 * self.n_layers if self.bidirectional else self.n_layers, x.size()[0], self.lstm_dim)).to(x.device)

        lstm_output, (h_n,c_n) = self.lstm(x_conv.permute(1,0,2), (h_0, c_0))
        lstm_output = lstm_output.permute(1,0,2)
        att = self.attention(lstm_output)
        hidden = torch.bmm(att.permute(0,2,1), lstm_output).mean(dim = 1)
        hidden = hidden.view(hidden.size()[0], -1)
        output = self.classifier(hidden)
        return output
    
    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = False, print_summary : bool = False, show_parent_layers : bool = False):
        sample = torch.zeros((1, self.seq_len, self.n_features), device = device)
        return print(summary(self, sample, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers))