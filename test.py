# use model as ViViT model or slowfast model
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.dataloader import VideoDataset
from src.utils.sampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pytorch_model_summary import summary

from src.utils.utility import video2tensor

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

    def forward(self, x : torch.Tensor)->torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViViT(nn.Module):
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
        super(ViViT, self).__init__()
        
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

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )

    def forward(self, x : torch.Tensor)->torch.Tensor:

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
        x = torch.sigmoid(x)

        return x

    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = True, print_summary : bool = True, show_parent_layers : bool = True)->None:
        sample = torch.zeros((1, self.n_frames, self.in_channels, self.image_size, self.image_size), device = device)
        return summary(self, sample, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers)



def generate_prob_curve(
    dataset : torch.Tensor, 
    model : torch.nn.Module, 
    batch_size : int = 32, 
    device : str = "cpu", 
    save_dir : Optional[str] = "./results/disruption_probs_curve.png",
    shot_list_dir : Optional[str] = "./dataset/KSTAR_Disruption_Shot_List.csv",
    shot_number : Optional[int] = None,
    clip_len : Optional[int] = None,
    dist_frame : Optional[int] = None,
    use_continuous_frame : bool = True
    ):
    prob_list = []
    video_len = dataset.size(0)

    model.to(device)
    model.eval()

    if video_len >= batch_size:
        batch_rest = video_len % batch_size
            
        for idx in range(int(video_len / batch_size)):
            with torch.no_grad():
                idx_start = batch_size * idx
                idx_end = batch_size * (idx + 1)

                frames = dataset[idx_start : idx_end, :, :, :, :]
                frames = frames.to(device)
                output = model(frames)
            
                prob_list.extend(
                    output.cpu().detach().numpy().reshape(-1,).tolist()
                )
        
        if batch_rest !=0:
            with torch.no_grad():
                idx_start = batch_size * (idx + 1)
                idx_end = idx_start + batch_rest

                frames = dataset[idx_start : idx_end, :, :, :, :]
                frames = frames.to(device)
                output = model(frames)
            
                prob_list.extend(
                    output.cpu().detach().numpy().reshape(-1,).tolist()
                )

    else:
        with torch.no_grad():
            frames = dataset[:, :, :, :, :]
            frames = frames.to(device)
            output = model(frames)
            prob_list.extend(
                output.cpu().detach().numpy().reshape(-1,).tolist()
            )
     
    if shot_list_dir and shot_number:
        shot_list = pd.read_csv(shot_list_dir)
        shot_info = shot_list[shot_list["shot"] == shot_number]
    else:
        shot_list = None
        shot_info = None

    if shot_info is not None:
        t_disrupt = shot_info["tTQend"].values[0]
        t_current = shot_info["tipminf"].values[0]
    else:
        t_disrupt = None
        t_current = None

    if use_continuous_frame:
        interval = 1
        # clip_len + distance만큼 외삽 진행
        prob_list = [0] * (clip_len + dist_frame) + prob_list
    else:
        interval = clip_len
        prob_list = [0] * (1 + int(dist_frame / clip_len)) + prob_list

    if save_dir:
        fps = 210

        time_x = np.arange(0, len(prob_list)) * (1/fps) * interval
        threshold_line = [0.5] * len(time_x)

        plt.figure(figsize = (8,5))
        plt.plot(time_x, threshold_line, 'k', label = "threshold(p = 0.5)")
        plt.plot(time_x, prob_list, 'b-', label = "disruption probs")

        if t_disrupt is not None:
            plt.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "thermal quench")
            print("thermal quench : {:.2f}".format(t_disrupt))
        
        if t_current is not None:
            plt.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "current quench")
            print("current quench : {:.2f}".format(t_current))

        plt.ylabel("probability")
        plt.xlabel("time(unit : s)")
        plt.ylim([0,1])
        plt.xlim([0,max(time_x)])
        plt.legend()
        plt.savefig(save_dir)

    return prob_list

if __name__ == "__main__":

    video_path = "./dataset/raw_videos/raw_videos/021273tv02.avi"
    shot_list_dir = "./dataset/KSTAR_Disruption_Shot_List.csv"
    shot_number = 21273

    dataset = video2tensor(
        dir = video_path,
        channels  = 3, 
        clip_len  = 21, 
        crop_size  = 224,
        resize_width  = 256,
        resize_height = 256,
        use_continuous_frame = False
    )

    print("dataset : ", dataset.size())

    # torch cuda initialize and clear cache
    torch.cuda.init()
    torch.cuda.empty_cache()

    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:1"
    else:
        device = 'cpu'

    model = ViViT(
        image_size = 224,
        patch_size = 16,
        n_frames = 21,
        dim = 64,
        depth = 4,
        n_heads = 4,
        pool = "cls",
        in_channels = 3,
        d_head = 64,
        dropout = 0.25,
        embedd_dropout=0.25,
        scale_dim = 4
    )

    weight = "./weights/ViViT_clip_21_dist_0_last.pt"
    model.load_state_dict(torch.load(weight))

    probs = generate_prob_curve(dataset, model, batch_size = 16, device = device, shot_list_dir = shot_list_dir, shot_number = shot_number, clip_len = 21, dist_frame = 0, use_continuous_frame = False)
    model.cpu()