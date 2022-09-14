import torch
import torch.nn as nn
from ConvLSTM import ConvLSTMEncoder
from ViViT import ViViTEncoder
from typing import Dict
from pytorch_model_summary import summary

class MultiModalModel(nn.Module):
    def __init__(self, n_classes : int, args_video : Dict, args_0D : Dict):
        super(MultiModalModel, self).__init__()
        self.n_classes = n_classes
        self.args_video = args_video
        self.args_0D = args_0D
        self.encoder_video = ViViTEncoder(**args_video)
        self.encoder_0D = ConvLSTMEncoder(**args_0D)
        linear_input_dims = self.encoder_0D.lstm_dim * 2 + self.encoder_video.dim
        self.classifier = nn.Sequential(
            nn.Linear(linear_input_dims, linear_input_dims // 2),
            nn.BatchNorm1d(linear_input_dims // 2),
            nn.ReLU(),
            nn.Linear(linear_input_dims // 2, n_classes)
        )
        
    def forward(self, x_video : torch.Tensor, x_0D : torch.Tensor)->torch.Tensor:
        x_video = self.encoder_video(x_video)
        x_0D = self.encoder_0D(x_0D)
        x = torch.cat([x_video, x_0D], axis = 1)
        output = self.classifier(x)
        return output
    
    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = False, print_summary : bool = True, show_parent_layers : bool = False):
        sample_video = torch.zeros((1, self.args_video["n_frames"], self.args_video["in_channels"], self.args_video["image_size"], self.args_video["image_size"]), device = device)
        sample_0D = torch.zeros((1, self.args_0D["seq_len"], self.args_0D["col_dim"]), device = device)
        return summary(self, sample_video, sample_0D, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers)


if __name__ == "__main__":
    
    args_video = {
        "image_size" : 128, 
        "patch_size" : 32, 
        "n_frames" : 21, 
        "dim": 64, 
        "depth" : 4, 
        "n_heads" : 8, 
        "pool" : 'cls', 
        "in_channels" : 3, 
        "d_head" : 64, 
        "dropout" : 0.25,
        "embedd_dropout":  0.25, 
        "scale_dim" : 4
    }
    
    args_0D = {
        "seq_len" : 21, 
        "col_dim" : 9, 
        "conv_dim" : 32, 
        "conv_kernel" : 3,
        "conv_stride" : 1, 
        "conv_padding" : 1,
        "lstm_dim" : 64, 
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model= MultiModalModel(2, args_video, args_0D)
    model.to(device)
    model.summary(device, True, False, True, False)