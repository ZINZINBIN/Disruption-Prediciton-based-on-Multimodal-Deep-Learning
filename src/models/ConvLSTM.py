import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from typing import List, Optional, Union, Tuple
from pytorch_model_summary import summary

class ConvLSTM(nn.Module):
    def __init__(
        self, 
        seq_len : int = 21, 
        col_dim : int = 10, 
        conv_dim : int = 32, 
        conv_kernel : int = 3,
        conv_stride : int = 1, 
        conv_padding : int = 1,
        lstm_dim : int = 64, 
        n_classes : int = 2, 
        mlp_dim : int = 64,
        ):
        
        super(ConvLSTM, self).__init__()
        self.col_dim = col_dim
        self.seq_len = seq_len
        self.lstm_dim = lstm_dim

        # spatio-conv encoder : analyze spatio-effect between variables
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = col_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim), 
            nn.ReLU(),
            nn.Conv1d(in_channels = conv_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim), 
            nn.ReLU(),
        )

        lstm_input_dim = self.compute_conv1d_output_dim(self.compute_conv1d_output_dim(seq_len, conv_kernel, conv_stride, conv_padding, 1), conv_kernel, conv_stride, conv_padding, 1)

        # temporl - lstm
        self.lstm = nn.LSTM(lstm_input_dim, lstm_dim, bidirectional = True, batch_first = False)
        self.w_s1 = nn.Linear(lstm_dim * 2, lstm_dim)
        self.w_s2 = nn.Linear(lstm_dim, lstm_dim)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_dim * 2, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, n_classes)
        )
    
    def compute_conv1d_output_dim(self, input_dim : int, kernel_size : int = 3, stride : int = 1, padding : int = 1, dilation : int = 1):
        return int((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def attention(self, lstm_output : torch.Tensor):
        attn_weight_matrix = self.w_s2(torch.tanh(self.w_s1(lstm_output)))
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim = 2)
        return attn_weight_matrix
    
    def encode(self, x:torch.Tensor):
        with torch.no_grad():
            x_conv = self.conv(x.permute(0,2,1))
            h_0 = Variable(torch.zeros(2, x.size()[0], self.lstm_dim)).to(x.device)
            c_0 = Variable(torch.zeros(2, x.size()[0], self.lstm_dim)).to(x.device)

            lstm_output, (h_n,c_n) = self.lstm(x_conv.permute(1,0,2), (h_0, c_0))
            lstm_output = lstm_output.permute(1,0,2)
            att = self.attention(lstm_output)
            hidden = torch.bmm(att.permute(0,2,1), lstm_output).mean(dim = 1)
            hidden = hidden.view(hidden.size()[0], -1)
            
        return hidden

    def forward(self, x : torch.Tensor):
        # x : (batch, seq_len, col_dim)
        x_conv = self.conv(x.permute(0,2,1))
        h_0 = Variable(torch.zeros(2, x.size()[0], self.lstm_dim)).to(x.device)
        c_0 = Variable(torch.zeros(2, x.size()[0], self.lstm_dim)).to(x.device)

        lstm_output, (h_n,c_n) = self.lstm(x_conv.permute(1,0,2), (h_0, c_0))
        lstm_output = lstm_output.permute(1,0,2)
        att = self.attention(lstm_output)
        hidden = torch.bmm(att.permute(0,2,1), lstm_output).mean(dim = 1)
        hidden = hidden.view(hidden.size()[0], -1)
        output = self.classifier(hidden)
        return output
    
    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = True, print_summary : bool = False, show_parent_layers : bool = False):
        sample = torch.zeros((1, self.seq_len, self.col_dim), device = device)
        return print(summary(self, sample, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers))

class ConvLSTMEncoder(nn.Module):
    def __init__(
        self, 
        seq_len : int = 21, 
        col_dim : int = 10, 
        conv_dim : int = 32, 
        conv_kernel : int = 3,
        conv_stride : int = 1, 
        conv_padding : int = 1,
        lstm_dim : int = 64, 
        ):
        
        super(ConvLSTMEncoder, self).__init__()
        self.col_dim = col_dim
        self.seq_len = seq_len
        self.lstm_dim = lstm_dim

        # spatio-conv encoder : analyze spatio-effect between variables
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = col_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim), 
            nn.ReLU(),
            nn.Conv1d(in_channels = conv_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim), 
            nn.ReLU(),
        )

        lstm_input_dim = self.compute_conv1d_output_dim(self.compute_conv1d_output_dim(seq_len, conv_kernel, conv_stride, conv_padding, 1), conv_kernel, conv_stride, conv_padding, 1)

        # temporl - lstm
        self.lstm = nn.LSTM(lstm_input_dim, lstm_dim, bidirectional = True, batch_first = False)
        self.w_s1 = nn.Linear(lstm_dim * 2, lstm_dim)
        self.w_s2 = nn.Linear(lstm_dim, lstm_dim)
    
    def compute_conv1d_output_dim(self, input_dim : int, kernel_size : int = 3, stride : int = 1, padding : int = 1, dilation : int = 1):
        return int((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def attention(self, lstm_output : torch.Tensor):
        attn_weight_matrix = self.w_s2(torch.tanh(self.w_s1(lstm_output)))
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim = 2)
        return attn_weight_matrix

    def forward(self, x : torch.Tensor):
        x_conv = self.conv(x.permute(0,2,1))
        h_0 = Variable(torch.zeros(2, x.size()[0], self.lstm_dim)).to(x.device)
        c_0 = Variable(torch.zeros(2, x.size()[0], self.lstm_dim)).to(x.device)

        lstm_output, (h_n,c_n) = self.lstm(x_conv.permute(1,0,2), (h_0, c_0))
        lstm_output = lstm_output.permute(1,0,2)
        att = self.attention(lstm_output)
        hidden = torch.bmm(att.permute(0,2,1), lstm_output).mean(dim = 1)
        hidden = hidden.view(hidden.size()[0], -1)
        return hidden
    
class ConvLSTMEncoderVer2(nn.Module):
    def __init__(
        self, 
        seq_len : int = 21, 
        col_dim : int = 10, 
        conv_dim : int = 32, 
        conv_kernel : int = 3,
        conv_stride : int = 1, 
        conv_padding : int = 1,
        lstm_dim : int = 64, 
        ):
        
        super(ConvLSTMEncoderVer2, self).__init__()
        self.col_dim = col_dim
        self.seq_len = seq_len
        self.lstm_dim = lstm_dim

        # spatio-conv encoder : analyze spatio-effect between variables
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = col_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim), 
            nn.ReLU(),
            nn.Conv1d(in_channels = conv_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim), 
            nn.ReLU(),
        )
        
        # temporl - lstm
        self.lstm = nn.LSTM(conv_dim, lstm_dim, bidirectional = True, batch_first = False)

    def forward(self, x : torch.Tensor):
        # x : (B, seq_len, col_dim)
        x_conv = self.conv(x.permute(0,2,1)) # (B, conv_dim, L_out : seq_len)
        h_0 = Variable(torch.zeros(2, x.size()[0], self.lstm_dim)).to(x.device)
        c_0 = Variable(torch.zeros(2, x.size()[0], self.lstm_dim)).to(x.device)

        lstm_output, (h_n,c_n) = self.lstm(x_conv.permute(2,0,1), (h_0, c_0))
        lstm_output = lstm_output.permute(1,0,2) # (B, lstm_dim * 2, L_out)
        hidden = lstm_output
        return hidden
    
    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = True, print_summary : bool = True, show_parent_layers : bool = True):
        sample = torch.zeros((1, self.seq_len, self.col_dim), device = device)
        return summary(self, sample, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers)

if __name__ == "__main__":
    # test
    time        = np.arange(0, 400, 0.1)
    amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))

    seq_len = 21
    col_len = 10
    batch_size = 32 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvLSTM(
        seq_len = seq_len,
        col_dim = col_len,
    )

    model.to(device)
    model.summary(device, True,  True, True, False)
    
    model.cpu()
    del model
    
    model = ConvLSTMEncoderVer2(
        seq_len = seq_len,
        col_dim = col_len
    )
    
    model.to(device)
    model.summary(device, True,  True, True, False)