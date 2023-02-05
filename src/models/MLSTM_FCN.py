''' Multivariate LSTM-FCNs for Time series classification
    Attention-LSTM based multivariate time series classification model
    The Convolution block with squeeze-and-excitation block is used for enhancement
    Reference
    - short summary : https://velog.io/@ddangchani/LSTM-FCN
    - paper : https://arxiv.org/pdf/1801.04503v2.pdf
    - code : https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py
    - papers-with-codes : https://paperswithcode.com/paper/multivariate-lstm-fcns-for-time-series
'''

import torch
import torch.nn as nn
from src.models.NoiseLayer import NoiseLayer
from pytorch_model_summary import summary

# Squeeze - Excite block
class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_channels : int, reduction : int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias = False),
            nn.Sigmoid()
        )
    
    def forward(self, x : torch.Tensor):
        B, C, T = x.size()
        x_new = self.avg_pool.forward(x).view(B,C)
        x_new = self.fc(x_new).view(B,C,1)
        return x * x_new.expand_as(x)
    
# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels:int, kernel_size : int, stride : int,alpha : float = 1.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(alpha)
        
    def forward(self, x : torch.Tensor):
        return self.relu(self.bn(self.conv(x)))
    
# Self-attention RNN module
class SelfAttentionRnn(nn.Module):
    def __init__(self, input_dim : int, hidden_dim : int, n_layers : int, bidirectional : bool = True, dropout : float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional = bidirectional, batch_first = False, num_layers = n_layers, dropout=dropout)
        
        if bidirectional:
            self.w_s1 = nn.Linear(hidden_dim * 2, hidden_dim)
            output_dim = hidden_dim * 2
        else:
            self.w_s1 = nn.Linear(hidden_dim, hidden_dim)
            output_dim = hidden_dim
        
        self.output_dim = output_dim
            
        self.w_s2 = nn.Linear(hidden_dim, hidden_dim)
    
    def attention(self, lstm_output : torch.Tensor):
        attn_weight_matrix = self.w_s2(torch.tanh(self.w_s1(lstm_output)))
        attn_weight_matrix = torch.nn.functional.softmax(attn_weight_matrix, dim = 2)
        return attn_weight_matrix
    
    def forward(self, x : torch.Tensor):
        # x : (batch, seq_len, col_dim)
        h_0 = torch.autograd.Variable(torch.zeros(2 * self.n_layers if self.bidirectional else self.n_layers, x.size()[0], self.hidden_dim)).to(x.device)
        c_0 = torch.autograd.Variable(torch.zeros(2 * self.n_layers if self.bidirectional else self.n_layers, x.size()[0], self.hidden_dim)).to(x.device)

        lstm_output, (h_n,c_n) = self.lstm(x.permute(1,0,2), (h_0, c_0))
        lstm_output = lstm_output.permute(1,0,2)
        att = self.attention(lstm_output)
        hidden = torch.bmm(att.permute(0,2,1), lstm_output).mean(dim = 1)
        hidden = hidden.view(hidden.size()[0], -1)
        return hidden
    

class MLSTM_FCN(nn.Module):
    def __init__(
        self, 
        n_features : int, 
        fcn_dim : int,
        kernel_size : int,
        stride : int,
        seq_len : int,
        lstm_dim : int,
        lstm_n_layers : int = 1,
        lstm_bidirectional:bool=True,
        lstm_dropout : float = 0.1,
        reduction : int = 16,
        alpha : float = 1.0, 
        n_classes : int = 2
        ):
        super().__init__()
        
        self.n_features = n_features
        self.seq_len = seq_len
        
        self.fcn = nn.Sequential(
            ConvBlock(n_features, fcn_dim, kernel_size, stride, alpha),
            SqueezeExciteBlock(fcn_dim, reduction),
            ConvBlock(fcn_dim, 2*fcn_dim, kernel_size, stride, alpha),
            SqueezeExciteBlock(2*fcn_dim, reduction),
        )
        
        self.noise = NoiseLayer(mean = 0, std = 1e-2)
        
        self.rnn = SelfAttentionRnn(n_features, lstm_dim, lstm_n_layers, lstm_bidirectional, lstm_dropout)
        
        feature_dims = self.rnn.output_dim + 2 * fcn_dim
        
        self.converter = nn.Linear(feature_dims, feature_dims)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims, feature_dims//2),
            nn.BatchNorm1d(feature_dims//2),
            nn.LeakyReLU(alpha),
            nn.Linear(feature_dims//2, n_classes)
        )
        
    def forward(self, x : torch.Tensor):
        # x : (N,T,C)
        x = self.noise(x)
        
        # RNN 
        x_rnn = self.rnn(x)
        
        # FCN
        x_fcn = self.shuffle(x) # (N, C, T)
        x_fcn = self.fcn(x_fcn).mean(axis = 2)

        # classifier
        x = torch.concat([x_rnn, x_fcn], axis = 1)
        x = self.converter(x)
        x = self.classifier(x)
    
        return x
    
    def encode(self, x : torch.Tensor):
        
        with torch.no_grad():
            
            x = self.noise(x)
            
            x_rnn = self.rnn(x)
        
            # FCN
            x_fcn = self.shuffle(x) # (N, C, T)
            x_fcn = self.fcn(x_fcn).mean(axis = 2)

            # classifier
            x = torch.concat([x_rnn, x_fcn], axis = 1)
            x = self.converter(x)
            
            return x        
    
    def shuffle(self, x : torch.Tensor):
        return x.permute(0,2,1)
    
    def summary(self):
        sample_x = torch.zeros((2, self.seq_len, self.n_features))
        summary(self, sample_x, batch_size = 2, show_input = True, print_summary=True)
        
if __name__ == "__main__":
    
    model = MLSTM_FCN(
        n_features = 11,
        fcn_dim=64,
        kernel_size=5,
        stride = 1,
        seq_len = 21,
        lstm_dim = 64,
        lstm_n_layers=2,
        lstm_bidirectional=True,
        lstm_dropout=0.1,
        reduction = 16,
        alpha = 1.0
    )

    model.summary()