from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pytorch_model_summary import summary

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

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x : (seq_len, batch_size, n_features)
        return x + self.pe[:x.size(0), :, :]

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class TStransformer(nn.Module):
    def __init__(self, n_features : int = 11, feature_dims : int = 256, max_len : int = 128, n_layers : int = 1, n_heads : int = 8, dim_feedforward : int = 1024, dropout : float = 0.1, cls_dims : int = 128, n_classes : int = 2):
        super(TStransformer, self).__init__()
        self.src_mask = None
        self.n_features = n_features
        self.max_len = max_len
        self.encoder_input_layer = nn.Linear(in_features = n_features, out_features = feature_dims)
        self.pos_enc = PositionalEncoding(d_model = feature_dims, max_len = max_len)
        self.encoder = nn.TransformerEncoderLayer(
            d_model = feature_dims, 
            nhead = n_heads, 
            dropout = dropout,
            dim_feedforward = dim_feedforward,
            activation = GELU()
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims, cls_dims),
            nn.BatchNorm1d(cls_dims),
            GELU(),
            nn.Linear(cls_dims, n_classes)
        )
        
    def encode(self, x: torch.Tensor)->torch.Tensor:
        with torch.no_grad():
            x = self.encoder_input_layer(x)
            x = x.permute(1,0,2)
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                device = x.device
                mask = self._generate_square_subsequent_mask(len(x)).to(device)
                self.src_mask = mask
        
            x = self.pos_enc(x)
            x = self.transformer_encoder(x, self.src_mask.to(x.device)).permute(1,0,2).mean(dim = 1)
        return x

    def forward(self, x : torch.Tensor)->torch.Tensor:
        x = self.encoder_input_layer(x)
        x = x.permute(1,0,2)
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask
        
        x = self.pos_enc(x)
        x = self.transformer_encoder(x, self.src_mask.to(x.device)).permute(1,0,2).mean(dim = 1) # (seq_len, batch, feature_dims)
        x = self.classifier(x)
        return x

    def _generate_square_subsequent_mask(self, size : int)->torch.Tensor:
        mask = (torch.triu(torch.ones(size,size))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def summary(self)->None:
        sample_x = torch.zeros((2, self.max_len, self.n_features))
        summary(self, sample_x, batch_size = 2, show_input = True, print_summary=True)