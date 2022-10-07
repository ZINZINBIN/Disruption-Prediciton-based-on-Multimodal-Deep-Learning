# video vision transformer model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pytorch_model_summary import summary

# Module for ViViT
# Residual module
class Residule(nn.Module):
    def __init__(self, fn):
        super(Residule, self).__init__()
        self.fn = fn
    def forward(self, x : torch.Tensor, **kwargs)->torch.Tensor:
        return self.fn(x, **kwargs) + x

# PreNorm module
class PreNorm(nn.Module):
    def __init__(self, dim : int, fn : Union[nn.Module, None]):
        super(PreNorm, self).__init__()
        self.dim = dim
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x : torch.Tensor, **kwargs)->torch.Tensor:
        return self.fn(self.norm(x), **kwargs)

# FeedForward module
class FeedForward(nn.Module):
    def __init__(self, dim : int, hidden_dim : int, dropout : float = 0.5):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x : torch.Tensor)->torch.Tensor:
        return self.net(x)

# Attention
class Attention(nn.Module):
    def __init__(self, dim : int, n_heads : int = 8, d_head : int = 64, dropout : float = 0.5):
        super(Attention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = d_head
        self.dropout = dropout
        self.att_mat = None

        project_out = not (n_heads == 1 and d_head == dim)

        self.inner_dim = d_head * n_heads
        self.scale = d_head ** (-0.5)

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias = False) # q, k, v : (dim, inner_dim) matrix

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x : torch.Tensor)->torch.Tensor:
        b, n, _, h = *x.shape, self.n_heads

        qkv = self.to_qkv(x)
        qkv = torch.chunk(qkv, 3, -1)

        q,k,v = map(lambda t : rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # obtain attention score
        attn = torch.softmax(dots, dim = -1)

        # multply attention score * value
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # rearrange as (b,n,h*d) => n_heads * d_head
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        out =  self.to_out(out)

        return out

class Transformer(nn.Module):
    def __init__(self, dim : int, depth : int, n_heads : int, d_head : int, mlp_dim : int, dropout : float = 0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=dim, n_heads = n_heads, d_head = d_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim=dim, hidden_dim = mlp_dim, dropout = dropout))
            ]))

    def forward(self, x : torch.Tensor):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViViT(nn.Module):
    def __init__(
        self, 
        image_size : int, 
        patch_size : int, 
        n_frames : int = 21, 
        n_classes : int = 2, 
        dim : int = 192, 
        depth : int = 4, 
        n_heads : int = 3, 
        pool : str = 'cls', 
        in_channels : int = 3, 
        d_head :int = 64, 
        dropout : float = 0.,
        embedd_dropout : float = 0., 
        scale_dim :int = 4, 
        ):
        super(ViViT, self).__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls(cls token) or mean(mean pooling)'
        assert image_size % patch_size == 0, "Image dimension(height and width) must be divisible by the patch_size"

        self.image_size = image_size
        self.n_frames = n_frames
        self.n_heads  = n_heads
        self.d_head = d_head
        self.depth = depth

        self.in_channels = in_channels

        n_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_frames, n_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(
            dim, depth, n_heads, d_head, dim * scale_dim, dropout
        )

        self.temporal_token = nn.Parameter(torch.randn(1,1,dim))
        self.temporal_transformer = Transformer(
            dim, depth, n_heads, d_head, dim * scale_dim, dropout
        )

        self.dropout = nn.Dropout(embedd_dropout)
        self.pool = pool
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x : torch.Tensor):

        if x.size()[1] == self.in_channels:
            x = torch.permute(x, (0,2,1,3,4))
            
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape
        cls_space_token = repeat(self.space_token, '() n d -> b t n d', b = b, t = t)
        cls_temporal_token = repeat(self.temporal_token, '() n d -> b n d', b = b)

        x = torch.cat((cls_space_token, x), dim = 2)
        x += self.pos_embedding[:,:,:(n+1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:,0], '(b t) ... -> b t ...', b = b)

        x = torch.cat((cls_temporal_token, x), dim = 1)
        x = self.temporal_transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp(x)

        return x
    
    def encode(self, x : torch.Tensor):
        with torch.no_grad():
            if x.size()[1] == self.in_channels:
                x = torch.permute(x, (0,2,1,3,4))
            
            x = self.to_patch_embedding(x)
            b, t, n, _ = x.shape
            cls_space_token = repeat(self.space_token, '() n d -> b t n d', b = b, t = t)
            cls_temporal_token = repeat(self.temporal_token, '() n d -> b n d', b = b)

            x = torch.cat((cls_space_token, x), dim = 2)
            x += self.pos_embedding[:,:,:(n+1)]
            x = self.dropout(x)

            x = rearrange(x, 'b t n d -> (b t) n d')
            x = self.space_transformer(x)
            x = rearrange(x[:,0], '(b t) ... -> b t ...', b = b)

            x = torch.cat((cls_temporal_token, x), dim = 1)
            x = self.temporal_transformer(x)
            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
            
        return x

    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = True, print_summary : bool = False, show_parent_layers : bool = False):
        sample = torch.zeros((1, self.n_frames, self.in_channels, self.image_size, self.image_size), device = device)
        return print(summary(self, sample, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers))


class ViViTEncoder(nn.Module):
    def __init__(
        self, 
        image_size : int, 
        patch_size : int, 
        n_frames : int, 
        dim : int = 192, 
        depth : int = 4, 
        n_heads : int = 3, 
        pool : str = 'cls', 
        in_channels : int = 3, 
        d_head :int = 64, 
        dropout : float = 0.,
        embedd_dropout : float = 0., 
        scale_dim :int = 4, 
        ):
        super(ViViTEncoder, self).__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls(cls token) or mean(mean pooling)'
        assert image_size % patch_size == 0, "Image dimension(height and width) must be divisible by the patch_size"

        self.image_size = image_size
        self.n_frames = n_frames
        self.in_channels = in_channels

        n_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_frames, n_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(
            dim, depth, n_heads, d_head, dim * scale_dim, dropout
        )

        self.temporal_token = nn.Parameter(torch.randn(1,1,dim))
        self.temporal_transformer = Transformer(
            dim, depth, n_heads, d_head, dim * scale_dim, dropout
        )

        self.dropout = nn.Dropout(embedd_dropout)
        self.pool = pool
        self.dim = dim


    def forward(self, x : torch.Tensor):

        if x.size()[1] == self.in_channels:
            x = torch.permute(x, (0,2,1,3,4))
            
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape
        cls_space_token = repeat(self.space_token, '() n d -> b t n d', b = b, t = t)
        cls_temporal_token = repeat(self.temporal_token, '() n d -> b n d', b = b)

        x = torch.cat((cls_space_token, x), dim = 2)
        x += self.pos_embedding[:,:,:(n+1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:,0], '(b t) ... -> b t ...', b = b)

        x = torch.cat((cls_temporal_token, x), dim = 1)
        x = self.temporal_transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return x

if __name__ == "__main__":

    model = ViViT(
        image_size = 128,
        patch_size = 32,
        n_classes = 2,
        n_frames = 21
    )

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = 'cpu'
    
    model.to(device)
    model.summary(device, show_input = True, show_hierarchical=True, print_summary=True, show_parent_layers=False)