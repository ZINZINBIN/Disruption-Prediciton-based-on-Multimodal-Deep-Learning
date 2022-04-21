'''
Lightweight Temporal Attention Encoder(L-TAE) for image time series
- Attention-based sequence encoding that maps a sequence of images to a single feature map
- A shared L-TAE is applied to all pixel positions of the image sequence
'''
import numpy as np
import torch
import torch.nn as nn
from typing import List

class PositionalEncoder(nn.Module):
    def __init__(self, d, T = 1000, repeat = None, offset = 0):
        super(PositionalEncoder, self).__init__()
        self.d = d 
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.update_location = False
    
    def forward(self, batch_position):
        if not self.updated_location:
            self.denom = self.denom.to(batch_position.device)
            self.updated_location = True
        sinusoid_table = (
            batch_position[:,:,None] / self.denom[None, None, :]
        )
        sinusoid_table[:,:,0::2] = torch.sin(sinusoid_table[:,:,0::2])
        sinusoid_table[:,:,1::2] = torch.cos(sinusoid_table[:,:,1::2])
        
        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim = -1
            ) 
        return sinusoid_table

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn

class LTAE(nn.Module):
    def __init__(self, in_channels : int = 128, n_head : int = 16, d_k : int = 4, hiddens : List[int] = [256,128], dropout : float = 0.2, d_model : int = 256, T : int = 1000, return_att : bool = False, positional_encoding = True):
        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.hiddens = hiddens
        self.return_att = return_att
        self.n_head = n_head
        self.d_k = d_k
        self.dropout = dropout
        self.d_model = d_model
        self.T = T
        self.positional_encoding = positional_encoding
        
        if d_model is not None:
            self.in_conv = nn.Conv1d(in_channels, d_model, 1)
        
        else:
            self.in_conv = None
        
        assert self.hiddens[0] == self.d_model
        
        if positional_encoding:
            self.positional_encoder = PositionalEncoder(d_model // n_head, T = T, repeat = n_head)
        else:
            self.positional_encoder = None
        
        self.attention_heads = MultiHeadAttention(
            n_head = n_head, d_k = d_k, d_in = self.d_model
        )
        
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels
        )
        
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=hiddens[-1]
        )
        
        layers = []
        
        for i in range(len(self.hiddens) - 1):
            layers.extend(
                [
                    nn.Linear(self.hiddens[i], self.hiddens[i+1]),
                    nn.BatchNorm1d(self.hidden[i+1]),
                    nn.ReLU()
                ]
            )
        
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x : torch.Tensor, batch_positions = None, pad_mask = None, return_comp = False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out