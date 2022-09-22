import torch
import torch.nn as nn
import math
from src.models.ConvLSTM import ConvLSTMEncoder, ConvLSTMEncoderVer2
from src.models.ViViT import ViViTEncoder
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
        x_video = x_video.mean(dim = 1) if self.encoder_video.pool == 'mean' else x_video[:, 0]
        x_0D = self.encoder_0D(x_0D)
        x = torch.cat([x_video, x_0D], axis = 1)
        output = self.classifier(x)
        return output
    
    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = False, print_summary : bool = True, show_parent_layers : bool = False):
        sample_video = torch.zeros((8,  self.args_video["in_channels"], self.args_video["n_frames"], self.args_video["image_size"], self.args_video["image_size"]), device = device)
        sample_0D = torch.zeros((8, self.args_0D["seq_len"], self.args_0D["col_dim"]), device = device)
        return summary(self, sample_video, sample_0D, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers)

# multi-modal deep learning
# we have to consider parameter sharing for two different data streams
# Video Frame Mapping with 0D data and fusion
class FusionNetwork(nn.Module):
    def  __init__(self, n_classes : int, args_video : Dict, args_0D : Dict,  args_fusion : Dict):
        super(FusionNetwork, self).__init__()
        
        self.args_video = args_video
        self.args_0D = args_0D
        self.args_fusion = args_fusion
        
        # single modality network as encoding feature representation
        self.encoder_video = ViViTEncoder(**args_video)
        self.encoder_0D = ConvLSTMEncoderVer2(**args_0D)
        
        assert self.encoder_0D.lstm_dim *2 == self.encoder_video.dim, "0D feature dims should be equal to video feature dims"
        seq_len = self.encoder_0D.seq_len
        feature_dims = self.encoder_0D.lstm_dim *  2
        
        # multimodal fusion for different feature representation
        self.fusion_module = nn.Sequential(
            nn.Conv1d(in_channels=feature_dims, out_channels=feature_dims, kernel_size = args_fusion['kernel_size'], stride = args_fusion['stride'], padding = 1),
            nn.BatchNorm1d(feature_dims),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(kernel_size = args_fusion['maxpool_kernel'], stride = args_fusion['maxpool_stride'], padding = 1)
        )
        
        classifier_dims = self.compute_fusion_dim(
            2 * seq_len, args_fusion['kernel_size'], args_fusion['stride']
        )
        
        classifier_dims = self.compute_fusion_dim(
            classifier_dims, args_fusion['maxpool_kernel'], args_fusion['maxpool_stride']
        )
        
        classifier_dims *= feature_dims
        
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dims, classifier_dims // 2),
            nn.BatchNorm1d(classifier_dims// 2),
            nn.ReLU(),
            nn.Linear(classifier_dims // 2, n_classes)
        )
        
    def compute_fusion_dim(self, input_dim : int, kernel_size : int = 3, stride : int = 1, padding : int = 1, dilation : int = 1):
        return math.floor((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        
    def forward(self, x_video : torch.Tensor, x_0D : torch.Tensor)->torch.Tensor:
        x_video = self.encoder_video(x_video)[:,1:,:]
        x_0D = self.encoder_0D(x_0D)
        b,t,d = x_video.size()       
        x = torch.zeros((b,2*t,d), device = x_video.device)
        x[:,0::2,:] = x_video
        x[:,1::2,:] = x_0D
        x  = self.fusion_module(x.permute(0,2,1)).view(x.size(0), -1)
        output = self.classifier(x)
        return output
    
    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = False, print_summary : bool = True, show_parent_layers : bool = False):
        sample_video = torch.zeros((8,  self.args_video["in_channels"], self.args_video["n_frames"], self.args_video["image_size"], self.args_video["image_size"]), device = device)
        sample_0D = torch.zeros((8, self.args_0D["seq_len"], self.args_0D["col_dim"]), device = device)
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
    
    args_fusion = {
        "kernel_size" : 4,
        "stride" : 2,
        "maxpool_kernel" : 3,
        "maxpool_stride" : 2,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model= MultiModalModel(2, args_video, args_0D)
    model.to(device)
    model.summary(device, True, False, True, False)
    
    del model
    
    model= FusionNetwork(2, args_video, args_0D, args_fusion)
    model.to(device)
    model.summary(device, True, False, True, False)
    