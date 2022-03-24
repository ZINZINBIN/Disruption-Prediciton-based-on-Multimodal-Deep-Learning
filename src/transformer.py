from re import M
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, max_len : int = 128):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len + 1, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1) # (max_len, 1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp() # (d_model/2, )

        pe[1:,0::2] = torch.sin(position * div_term)
        pe[1:,1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, doy):
        return self.pe[doy, :]

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model : int, d_ff : int, dropout : float = 0.1):
        super(PositionWiseFeedForward,  self).__init__()
        self.w_1 = nn.Linear(d_model,  d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNormCustom(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNormCustom, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x:torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SubLayerConnection(nn.Module):
    def __init__(self, size : int, dropout : float = 0.1):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size) # SITS-BERT에서는 LayerNorm을 자체 정의함..
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:torch.Tensor, sublayer:nn.Module)->torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class Attention(nn.Module):

    def forward(self, query : torch.Tensor, key : torch.Tensor, value : torch.Tensor, mask = None, dropout = None):
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = F.softmax(scores, dim = 1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h : int, d_model : int, dropout = 0.1):
        super(MultiHeadAttention,  self).__init__()
        assert d_model % h == 0
        
        self.d_k = d_model / h
        self.h = h
        self.d_model = d_model

        self.linear_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(3)
        ])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query : torch.Tensor, key : torch.Tensor, value : torch.Tensor, mask = None):
        batch_size = query.size(0)

        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linear_layers, (query, key, value))
        ]

        x, attn = self.attention(query, key, value, mask = mask, dropout = self.dropout)

        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class TransformerBlock(nn.Module):
    def __init__(self, hidden : int, attn_heads : int, feed_forward_hidden : int, dropout : float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(h = attn_heads, d_model = hidden)
        self.feed_forward = PositionWiseFeedForward(d_model = hidden, d_ff = feed_forward_hidden, dropout = dropout)
        self.input_sublayer = SubLayerConnection(size = hidden, dropout = dropout)
        self.output_sublayer = SubLayerConnection(size = hidden, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor, mask : Optional[torch.Tensor]):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x,_x,_x, mask = mask))
        x  = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class BertEmbedding(nn.Module):
    def __init__(self, num_features : int, embedding_dim : int, max_len : int = 128, dropout :float = 0.1):
        super(BertEmbedding, self).__init__()
        self.input = nn.Linear(num_features, embedding_dim)
        self.position = PositionalEncoding(d_model = embedding_dim, max_len = max_len)
        self.dropout = nn.Dropout(p = dropout)
        self.embed_size = embedding_dim

    def forward(self, input_sequence : torch.Tensor, doy_sequence : torch.Tensor):
        batch_size = input_sequence.size(0)
        seq_length = input_sequence.size(1)

        obs_embed = self.input(input_sequence) # batch_size, seq_length, embedding_dim
        x = obs_embed.repeat(1,1,2) # batch_size, seq_length, embedding_dim * 2
        
        for i in range(batch_size):
            x[i, :, self.embed_size : ] = self.position(doy_sequence[i, :]) 

        return self.dropout(x)

class SBERT(nn.Module):
    def __init__(self, num_features : int, hidden  : int, n_layers : int, attn_heads : int, max_len : int = 128, dropout : float = 0.1):
        super(SBERT, self).__init__()
        self.hidden = hidden
        self.num_features = num_features
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.dropout = dropout
        self.max_len = max_len

        self.feed_forward_hidden = hidden * 4
        self.embedding = BertEmbedding(num_features, int(hidden/2), max_len = max_len, dropout = dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)
        ])

    def forward(self, x : torch.Tensor, doy : torch.Tensor, mask : Optional[torch.Tensor]):
        mask = (mask > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(input_sequence = x, doy_sequence = doy)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

class MulticlassClassifier(nn.Module):
    def __init__(self, enc_dims : int, hidden : int, num_classes : int = 2, alpha : float = 0.01):
        super(MulticlassClassifier, self).__init__()
        self.enc_dims = enc_dims
        self.pooling = nn.MaxPool1d(enc_dims)
        pooling_dims = self.get_pooling_dims()
        self.mlp = nn.Sequential(
            nn.Linear(pooling_dims, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(alpha),
            nn.Linear(hidden, num_classes)
        )

    def get_pooling_dims(self):
        sample = torch.zeros((1, self.enc_dims))
        sample_output = self.pooling(sample)
        return sample_output.size(1)

    def forward(self, x : torch.Tensor, mask : Optional[torch.Tensor] = None):
        x = self.pooling(x.permute(0,2,1)).squeeze()
        x = self.mlp(x)
        return x

class SBERTDisruptionClassifier(nn.Module):
    def __init__(self, sbert : SBERT, mlp_hidden : int, num_classes : int = 2):
        super(SBERTDisruptionClassifier, self).__init__()
        self.sbert = sbert
        enc_dims = self.get_sbert_output()
        self.classifier = MulticlassClassifier(enc_dims, mlp_hidden, num_classes)

    def get_sbert_output(self):
        seq_len = self.sbert.max_len
        num_features = self.sbert.num_features
        doy_dims = seq_len

        sample_x = torch.zeros((1, doy_dims))
        sample_doy = torch.zeros((1, seq_len, num_features))
        sample_mask = None
        sample_output = self.sbert(sample_x, sample_doy, sample_mask)

        return sample_output.size(1)

    def forward(self, x : torch.Tensor, doy : torch.Tensor, mask : Optional[torch.Tensor]):
        # doy : (batch_size, doy_dims)
        # x : (batch_size, seq_len, num_features)
        x = self.sbert(x, doy, mask)
        return self.classifier(x, mask)