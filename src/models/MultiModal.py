import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from src.models.ConvLSTM import ConvLSTM, ConvLSTMEncoder, ConvLSTMEncoderVer2
from src.models.ViViT import ViViTEncoder, ViViT
from typing import Dict, Literal
from pytorch_model_summary import summary

# Simple case : fusion with concatenating each latent vector
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
        
    def forward(self, x_video : torch.Tensor, x_0D : torch.Tensor):
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

# MultiModal Network for GradientBlending
class MultiModalNetwork(nn.Module):
    def __init__(self, n_classes : int, args_video : Dict, args_0D : Dict, use_stream : Literal["video","0D","multi", "multi-GB"] = "multi-GB"):
        super(MultiModalNetwork, self).__init__()
        self.n_classes = n_classes
        self.args_video = args_video
        self.args_0D = args_0D
        self.vis_model = ViViT(**args_video)
        self.ts_model = ConvLSTM(**args_0D)        
        
        linear_input_dims = self.ts_model.lstm_dim * 2 + self.vis_model.dim
        
        self.classifier = nn.Sequential(
            nn.Linear(linear_input_dims, linear_input_dims // 2),
            nn.BatchNorm1d(linear_input_dims // 2),
            nn.ReLU(),
            nn.Linear(linear_input_dims // 2, n_classes)
        )
        
        self.vis_latent = None
        self.ts_latent = None
        
        self.use_stream = use_stream
        
        if use_stream == 'video':
            self.ts_model.training = False
            self.classifier.training = False
        elif use_stream == '0D':
            self.vis_model.training = False
            self.classifier.training = False
        else:
            self.ts_model.training = True
            self.vis_model.training = True
            self.classifier.training = True
        
        if use_stream == "multi" or use_stream == "multi-GB":
            self.vis_hook = self.vis_model.mlp[0].register_forward_hook(self.get_vis_latent)
            self.ts_hook = self.ts_model.classifier[0].register_forward_hook(self.get_ts_latent)    
        
    def remove_my_hooks(self):
        self.vis_hook.remove()
        self.ts_hook.remove()
        
    def update_use_stream(self, use_stream : Literal["video","0D","multi"]):
        self.use_stream = use_stream
        
        if use_stream == 'video':
            self.ts_model.training = False
            self.vis_model.training = True
            self.classifier.training = False
        elif use_stream == '0D':
            self.ts_model.training = True
            self.vis_model.training = False
            self.classifier.training = False
        else:
            self.ts_model.training = True
            self.vis_model.training = True
            self.classifier.training = True
        
        if use_stream == "multi" or use_stream == "multi-GB":
            self.vis_hook = self.vis_model.mlp[0].register_forward_hook(self.get_vis_latent)
            self.ts_hook = self.ts_model.classifier[0].register_forward_hook(self.get_ts_latent)    
     
    def get_vis_latent(self, module:nn.Module, input : torch.Tensor, output : torch.Tensor):
        self.vis_latent = input
    
    def get_ts_latent(self, module:nn.Module, input : torch.Tensor, output : torch.Tensor):
        self.ts_latent = input
        
    def forward(self, x_vis : torch.Tensor, x_ts : torch.Tensor):
        return self.forward_stream(x_vis, x_ts)
    
    def forward_stream(self, x_vis : torch.Tensor, x_ts : torch.Tensor):
        if self.use_stream == "video":
            out_vis = self.vis_model(x_vis)
            return out_vis
        
        elif self.use_stream == "0D":
            out_ts = self.ts_model(x_ts)
            return out_ts
        
        else:
            out_vis = self.vis_model(x_vis)
            out_ts = self.ts_model(x_ts)
            
            vis_latent = self.vis_latent[0]
            ts_latent = self.ts_latent[0]
            x = torch.cat([vis_latent, ts_latent], axis = 1)
            out_multi = self.classifier(x)
    
            return out_multi if self.use_stream == 'multi' else (out_multi, out_vis, out_ts)
            
    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = False, print_summary : bool = True, show_parent_layers : bool = False):
        sample_video = torch.zeros((8,  self.args_video["in_channels"], self.args_video["n_frames"], self.args_video["image_size"], self.args_video["image_size"]), device = device)
        sample_0D = torch.zeros((8, self.args_0D["seq_len"], self.args_0D["col_dim"]), device = device)
        return summary(self, sample_video, sample_0D, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers)

# Tensor Fusion Network
# reference paper : https://arxiv.org/pdf/1707.07250.pdf
# reference code: https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py
# In this code, we also use Gradient Blending Method 

class TensorFusionNetwork(nn.Module):
    def __init__(self, n_classes : int, args_video : Dict, args_0D : Dict):
        super(TensorFusionNetwork, self).__init__()
        self.n_classes = n_classes
        self.args_video = args_video
        self.args_0D = args_0D
        
        # Modality Embedding SubNetwork
        self.embedd_subnet = nn.ModuleDict({
            "network_video" : ViViT(**args_video),
            "network_0D" : ConvLSTM(**args_0D)
            })
        
        self.network_0D_dims = self.embedd_subnet['network_0D'].lstm_dim * 2
        self.network_video_dims = self.embedd_subnet['network_video'].dim
        
        self.fusion_input_dims = (self.network_0D_dims + 1) * (self.network_video_dims + 1)

        # Tensor Fusion Layer as classifier
        self.dropout = nn.Dropout(0)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_input_dims, self.fusion_input_dims // 2),
            nn.BatchNorm1d(self.fusion_input_dims // 2),
            nn.ReLU(),
            nn.Linear(self.fusion_input_dims // 2, n_classes)
        )
        
        self.h_vis = None
        self.h_0D = None
        
        self.vis_hook = self.embedd_subnet['network_video'].mlp[0].register_forward_hook(self.get_vis_latent)
        self.ts_hook = self.embedd_subnet['network_0D'].classifier[0].register_forward_hook(self.get_ts_latent)    
        
    def remove_my_hooks(self):
        self.vis_hook.remove()
        self.ts_hook.remove()
        
    def get_vis_latent(self, module:nn.Module, input : torch.Tensor, output : torch.Tensor):
        self.h_vis = input
    
    def get_ts_latent(self, module:nn.Module, input : torch.Tensor, output : torch.Tensor):
        self.h_0D = input

    def forward(self, x_vis : torch.Tensor, x_0D : torch.Tensor):
        out_vis = self.embedd_subnet['network_video'](x_vis)
        out_0D = self.embedd_subnet['network_0D'](x_0D)
        
        h_vis = self.h_vis[0]
        h_0D = self.h_0D[0]
        
        batch_size = h_vis.size()[0]
        
        if h_vis.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _h_vis = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), h_vis), dim=1)
        _h_0D = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), h_0D), dim=1)
    
        fusion_tensor = torch.bmm(_h_vis.unsqueeze(2), _h_0D.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(batch_size, -1)
        
        out = self.classifier(self.dropout(fusion_tensor))
  
        return (out, out_vis, out_0D)

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
        
    def forward(self, x_video : torch.Tensor, x_0D : torch.Tensor):
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
    
    del model