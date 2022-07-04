import numpy as np
import torch
import torch.nn as nn
import os, random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm
from typing import Optional
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from pytorch_model_summary import summary

# image augmentation
train_transforms = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
)

valid_transforms = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
)


# class CatsDogsDataset(Dataset):


# PatchEmbedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels : int = 3, patch_size : int = 16, embedd_size : int = 768, img_size : int = 224):
        super(PatchEmbedding, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedd_size = embedd_size
        self.img_size = img_size

        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1 = patch_size, s2 = patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embedd_size)
        )

        # class token 
        self.cls_token = nn.Parameter(torch.randn(1,1,embedd_size))

        # position encoding
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, embedd_size))

    def forward(self, x : torch.Tensor)->torch.Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_token = repeat(self.cls_token, '() n e -> b n e', b = b)

        # add cls_token
        x = torch.cat([cls_token, x], dim = 1) # z = [z0 : cls token, z_p1, z_p2, ... z_pn]

        # position encoding
        x += self.positions

        return x

# Multi-Head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embedd_size : int = 768, n_heads : int = 8, dropout : float = 0.5):
        super(MultiHeadAttention, self).__init__()
        self.embedd_size = embedd_size
        self.n_heads = n_heads
        self.dropout = dropout

        self.keys = nn.Linear(embedd_size, embedd_size)
        self.queries = nn.Linear(embedd_size, embedd_size)
        self.values = nn.Linear(embedd_size, embedd_size)

        self.att_drops = nn.Dropout(dropout)
        self.projection = nn.Linear(embedd_size, embedd_size)
        self.scaling = (embedd_size // n_heads)  

    def forward(self, x : torch.Tensor, mask : Optional[torch.Tensor] = None)->torch.Tensor:
        q = rearrange(self.queries(x), 'b n (h d) -> b h n d', h = self.n_heads)
        k = rearrange(self.keys(x), 'b n (h d) -> b h n d', h = self.n_heads)
        v = rearrange(self.values(x), 'b n (h d) -> b h n d', h = self.n_heads)  

        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k) 

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(-mask, fill_value)    

        att = F.softmax(energy, dim = -1) * self.scaling
        att = self.att_drops(att)

        out = torch.einsum('bhal, bhlv -> bhav', att, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)

        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, embedd_size : int, expansion : int = 4, drop_p : float = 0.1):
        super(FeedForwardBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(embedd_size, expansion * embedd_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * embedd_size, embedd_size)
        )
    
    def forward(self , x : torch.Tensor)->torch.Tensor:
        return self.layer(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedd_size : int = 768, drop_p : float = 0.1, forward_expansion : int = 4, forward_drop_p : float = 0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__()
        self.embedd_size = embedd_size
        self.drop_p = drop_p
        self.forward_expansion = forward_expansion
        self.forward_drop_p = forward_drop_p

        fn = nn.Sequential(
            nn.LayerNorm(embedd_size),
            MultiHeadAttention(embedd_size, **kwargs),
            nn.Dropout(drop_p)
        )

        self.sub_block_1 = ResidualAdd(fn)

        self.sub_block_2 = nn.Sequential(
            nn.LayerNorm(embedd_size),
            FeedForwardBlock(embedd_size, forward_expansion, forward_drop_p),
            nn.Dropout(drop_p)
        )

    def forward(self, x : torch.Tensor, **kwargs):
        x = self.sub_block_1(x, **kwargs)
        x = self.sub_block_2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth : int = 12, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.layer = nn.Sequential()

        for idx, layer in enumerate([TransformerEncoderBlock(**kwargs) for _ in range(depth)]):
            self.layer.add_module("Transformer_Encoder_Block_layer_"+str(idx+1), layer)
        
    def forward(self, x: torch.Tensor, **kwargs)->torch.Tensor:
        return self.layer(x, **kwargs)

class ClassificationHead(nn.Module):
    def __init__(self, embedd_size : int = 768, n_classes : int = 1000):
        super(ClassificationHead, self).__init__()
        self.embedd_size = embedd_size
        self.n_classes = n_classes

        self.layer = nn.Sequential(
            Reduce('b n e -> b e', reduction = "mean"),
            nn.LayerNorm(embedd_size),
            nn.Linear(embedd_size, n_classes)
        )

    def forward(self, x : torch.Tensor)->torch.Tensor:
        return self.layer(x)

class ViT(nn.Module):
    def __init__(self, in_channels : int = 3, patch_size : int = 16, embedd_size : int = 768, img_size : int = 224, depth : int = 12, n_classes : int = 1000, **kwargs):
        super(ViT, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedd_size = embedd_size
        self.img_size = img_size
        self.depth = depth
        self.n_classes = n_classes

        self.layer = nn.Sequential(
            PatchEmbedding(in_channels, patch_size, embedd_size, img_size),
            TransformerEncoder(depth, embedd_size = embedd_size, **kwargs),
            ClassificationHead(embedd_size, n_classes)
        )

    def forward(self, x : torch.Tensor)->torch.Tensor:
        return self.layer(x)
    
    def summary(self):
        samples = torch.zeros((1,3,224,224), device = "cpu")
        summary(self, samples, show_input = True, show_hierarchical=False, print_summary=True, show_parent_layers=True, max_depth = None)
    

if __name__ == "__main__":
    model = ViT()
    model.summary()