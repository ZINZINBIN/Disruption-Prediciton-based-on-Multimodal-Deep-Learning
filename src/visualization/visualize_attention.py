# heat map for ViViT model
# visualize attention map for each frame
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Literal

# transform : crop and normalize
def crop(buffer, original_height, original_width, crop_size):
    mid_x, mid_y = original_height // 2, original_width // 2
    offset_x, offset_y = crop_size // 2, crop_size // 2
    buffer = buffer[:, mid_x - offset_x:mid_x+offset_x, mid_y - offset_y: mid_y+ offset_y, :]
    return buffer

def normalize(buffer:np.ndarray):
    for i, frame in enumerate(buffer):
        frame -= np.array([[[90.0, 98.0, 102.0]]])
        buffer[i] = frame
    return buffer

# Vision Transformer Attention Rollout method
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from src.models.ViViT import ViViT

class ViViTAttentionRollout:
    def __init__(self, model : ViViT, layer_name = '0.fn.to_qkv', head_fusion = 'mean', discard_ratio = 0.9, transformer : Literal['temporal', 'space'] = 'space'):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        
        if transformer == 'space':
            module_name = "space_transformer"
        else:
            module_name = "temporal_transformer"
        
        self.module_name = module_name
    
        for name, module in model.named_modules():
            if layer_name in name and module_name in name:
                module.register_forward_hook(self.get_attention)
        
        self.attentions = []
    
    def get_attention(self, module:nn.Module, input:torch.Tensor, output:torch.Tensor):
        
        h = self.model.n_heads
        output = torch.chunk(output, 3, -1)
        q,k,v = map(lambda t : rearrange(t, 'b n (h d) -> b h n d', h = h), output)
        
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.model.d_head ** (-0.5)
        attn = torch.softmax(dots, dim = -1)
        self.attentions.append(attn.cpu())
    
    def __call__(self, input_tensor : torch.Tensor):
        self.attentions = []
        seq_len = self.model.n_frames
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
        if self.module_name == "space_transformer":
            return spatio_rollout(self.attentions, self.discard_ratio, self.head_fusion, seq_len)
        else:
            return temporal_rollout(self.attentions, self.discard_ratio, self.head_fusion)

# for space transformer : compute attention matrix with spatio - correlation
def spatio_rollout(attentions : List[torch.Tensor], discard_ratio : float, head_fusion : Literal["mean", 'max', 'min'], seq_len : int = 21,):
    result = torch.eye(attentions[0].size()[-1]).unsqueeze(0)
    result = result.repeat((seq_len,1,1))
    
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            
            # attention_heads_fused : (21, 1, 17, 17)
            flat = attention_heads_fused.view(attention_heads_fused.size()[0], -1)
            
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), dim = -1, largest = False)
            indices = indices[indices != 0]
            flat[0, indices] = 0
            
            I = torch.eye(attention.size(-1)).unsqueeze(0).repeat((seq_len,1,1))
            a = (attention_heads_fused + 1.0 * I) / 2.0
            
            result = torch.bmm(a, result)
            
    mask = result[:,0,1:]
    width = int(mask.size()[-1] ** 0.5)
    mask = mask.numpy().reshape(result.size()[0], width, width)
    mask = mask / np.max(mask)
    return mask

# for teporal transformer : compute attention matrix with temporal - correlation
def temporal_rollout(attentions : List[torch.Tensor], discard_ratio : float, head_fusion : Literal["mean", 'max', 'min']):
    result = torch.eye(attentions[0].size()[-1]).unsqueeze(0)
    with torch.no_grad():
        for attention in attentions:
            
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            
            # attention_heads_fused : (1, 22, 22)
            flat = attention_heads_fused.view(attention_heads_fused.size()[0], -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), dim = -1, largest = False)
            indices = indices[indices != 0]
            flat[0, indices] = 0
            
            I = torch.eye(attention.size(-1)).unsqueeze(0)
            a = (attention_heads_fused + 1.0 * I) / 2.0
            result = torch.bmm(a, result)
    
    mask = result[:,1:,1:].squeeze(0)
    mask = mask.numpy()
    mask = mask / np.max(mask)
    return mask


# visualize attention rollout with image sequence
def visualize_spatio_attention(shot : np.ndarray, att_map : np.ndarray, size : int = 128, save_dir = "./results/spatio_attention.png"):
    # shot : (h, w, c) [ex : (128,128,3)]
    # att_mask : (n_h, n_w) [ex : (4,4), n_h : h // patch, n_w : w // patch]
    att_map = cv2.resize(att_map, (128,128))
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(shot)
    _ = ax2.imshow(att_map)
    plt.savefig(save_dir)
    
def visualize_temporal_attention(att_map : np.ndarray, save_dir : str = "./result/temporal_attention.png"):
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.set_title('Temporal attention mask')
    plt.imshow(att_map)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.savefig(save_dir)