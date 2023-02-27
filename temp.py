from src.models.MultiModal import MultiModalModel, MultiModalModel_GB
from torch.utils.data import DataLoader
import os, torch

if(torch.cuda.device_count() >= 1):
    device = "cuda:1"
else:
    device = 'cpu'
    
ts_cols = [
    '\\q95', '\\ipmhd', '\\kappa', '\\tritop', '\\tribot',
    '\\betap','\\betan','\\li', '\\WTOT_DLM03', '\\ne_inter01', 
    '\\TS_NE_CORE_AVG', '\\TS_TE_CORE_AVG'
]

args_video = {
    "image_size" : 128, 
    "patch_size" : 16, 
    "n_frames" : 21, 
    "dim": 128, 
    "depth" : 2, 
    "n_heads" : 8, 
    "pool" : 'cls', 
    "in_channels" : 3, 
    "d_head" : 64, 
    "dropout" : 0.25,
    "embedd_dropout": 0.25, 
    "scale_dim" : 8,
}

args_0D = {
    "n_features" : len(ts_cols), 
    "feature_dims" : 128,
    "max_len" : 21, 
    "n_layers" : 4,
    "n_heads" : 8,
    "dim_feedforward":512, 
    "dropout" : 0.25,
}

model = MultiModalModel(
    2,
    21,
    args_video,
    args_0D
)

model.to(device)

from src.utils.utility import measure_computation_time_multi
t_avg, t_std, t_measures = measure_computation_time_multi(model, input_shape_vis = (1, 3, 21, 128, 128), input_shape_0D = (1,21,12), n_samples = 16, device = device)

print("t_avg : {:.3f}, t_std : {:.3f}".format(t_avg, t_std))