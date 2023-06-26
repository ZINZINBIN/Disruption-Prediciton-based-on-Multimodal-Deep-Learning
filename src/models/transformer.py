from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pytorch_model_summary import summary
from src.models.NoiseLayer import NoiseLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, max_len : int = 128):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1) # (max_len, 1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp() # (d_model // 2, )

        pe[:,0::2] = torch.sin(position * div_term)

        if d_model % 2 != 0:
            pe[:,1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0,1) # shape : (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor):
        # x : (seq_len, batch_size, n_features)
        return x + self.pe[:x.size(0), :, :]

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        
class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        n_features : int = 11, 
        kernel_size : int = 3,
        feature_dims : int = 256, 
        max_len : int = 128, 
        n_layers : int = 1, 
        n_heads : int = 8, 
        dim_feedforward : int = 1024, 
        dropout : float = 0.1, 
        ):
        super(TransformerEncoder, self).__init__()
        self.src_mask = None
        self.n_features = n_features
        self.max_len = max_len
        self.feature_dims = feature_dims
        
        self.noise = NoiseLayer(mean = 0, std = 1e-3)
        
        if kernel_size // 2 == 0:
            print("kernel sholud be odd number")
            kernel_size += 1
        padding = (kernel_size - 1) // 2
        
        self.filter = nn.Sequential(
            nn.Conv1d(in_channels = n_features, out_channels = feature_dims, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.Conv1d(in_channels = feature_dims, out_channels = feature_dims, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.BatchNorm1d(feature_dims),
            nn.ReLU(),
        )
        
        self.pos_enc = PositionalEncoding(d_model = feature_dims, max_len = max_len)
        
        encoder = nn.TransformerEncoderLayer(
            d_model = feature_dims, 
            nhead = n_heads, 
            dropout = dropout,
            dim_feedforward = dim_feedforward,
            activation = GELU()
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder, num_layers=n_layers)
        self.connector = nn.Sequential(
            nn.Linear(feature_dims, feature_dims),
            nn.LayerNorm(feature_dims),
            nn.GELU()
        )
        
    def forward(self, x : torch.Tensor):
        # Transformer Encoder
        # Embedding for low dimensional space
        x = self.noise(x)
        
        # convolution process
        x = self.filter(x.permute(0,2,1)).permute(0,2,1)
        
        # mapping for transformer module
        x = x.permute(1,0,2)
        
        self.src_mask = self._generate_square_subsequent_mask(len(x), x.device)
        
        x = self.pos_enc(x)
        x = self.transformer_encoder(x, self.src_mask).permute(1,0,2).mean(dim = 1) # (seq_len, batch, feature_dims)
        x = self.connector(x)
        return x

    def _generate_square_subsequent_mask(self, size : int, device : str):
        mask = (torch.triu(torch.ones(size,size))==1).to(device).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def summary(self):
        sample_x = torch.zeros((1, self.max_len, self.n_features))
        summary(self, sample_x, batch_size = 1, show_input = True, print_summary=True)

class Transformer(nn.Module):
    def __init__(
        self, 
        n_features : int = 11, 
        kernel_size : int = 5,
        feature_dims : int = 256, 
        max_len : int = 128, 
        n_layers : int = 1, 
        n_heads : int = 8, 
        dim_feedforward : int = 1024, 
        dropout : float = 0.1, 
        cls_dims : int = 128, 
        n_classes : int = 2
        ):
        super(Transformer, self).__init__()
        self.max_len = max_len
        self.n_features = n_features
        self.encoder = TransformerEncoder(n_features, kernel_size, feature_dims, max_len, n_layers, n_heads, dim_feedforward, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims, cls_dims),
            nn.LayerNorm(cls_dims),
            GELU(),
            nn.Linear(cls_dims, n_classes)
        )
        
    def encode(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.encoder(x)
        return x

    def forward(self, x : torch.Tensor):
        # Transformer Encoder
        x = self.encoder(x)
        # Disruption classifier
        x = self.classifier(x)
        return x
    
    def summary(self):
        sample_x = torch.zeros((1, self.max_len, self.n_features))
        summary(self, sample_x, batch_size = 1, show_input = True, print_summary=True)