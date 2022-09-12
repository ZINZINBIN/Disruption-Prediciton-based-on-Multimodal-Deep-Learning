# heat map for ViViT model
# visualize attention map for each frame
import torch
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


def rollout(attentions : List[torch.Tensor], discard_ratio : float, head_fusion : Literal['mean', 'max', 'min']):
    # attentions : List which consist of [1, channels, height, width] size of attention data
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == 'mean':
                attention_heads_fused = attention.mean(axis = 1)
            elif head_fusion == 'max':
                attention_heads_fused = attention.max(axis = 1)[0]
            elif head_fusion == 'min':
                attention_heads_fused = attention.min(axis = 1)[0]
            
            flat = attention_heads_fused.view(attention_heads_fused.size()[0], -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), dim = -1, largest = False)
            indices = indices[indices != 0]
            flat[0, indices] = 0
            
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2.0
            a = a / a.sum(dim = -1)
            
            result = torch.matmul(a, result)
    
    mask = result[0,0,1:]
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask

def add_mask_on_img_sequence(buffer: np.ndarray, masks : np.ndarray, h : int = 256, w : int = 256, crop_size : int = 128, seq_len : int = 21):
    
    buffer = crop(buffer, h, w, crop_size)
    buffer = normalize(buffer)
    
    for seq_idx in range(seq_len):
        img = buffer[seq_idx, :, : ,:]
        mask = masks[seq_idx, :, :, :]
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        buffer[seq_idx, :, :, :] = cam
    
    return buffer
        
        