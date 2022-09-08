import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from typing import List, Optional, Union, Tuple

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

    def attention(self, lstm_output : torch.Tensor)->torch.Tensor:
        attn_weight_matrix = self.w_s2(torch.tanh(self.w_s1(lstm_output)))
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim = 2)
        return attn_weight_matrix
    
    def encode(self, x:torch.Tensor)->torch.Tensor:
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

    def forward(self, x : torch.Tensor)->torch.Tensor:
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

    sample_data = torch.zeros((batch_size, seq_len, col_len)).to(device)
    sample_output = model(sample_data)

    print("sample_data : ", sample_data.size())
    print("sample_output : ", sample_output.size())