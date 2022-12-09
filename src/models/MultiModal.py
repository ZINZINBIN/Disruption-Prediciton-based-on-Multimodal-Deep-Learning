import torch
import torch.nn as nn
from torch.autograd import Variable
from src.models.ConvLSTM import ConvLSTM, ConvLSTMEncoder
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
class MultiModalModel_GB(nn.Module):
    def __init__(self, n_classes : int, args_video : Dict, args_0D : Dict, use_stream : Literal["video","0D","multi", "multi-GB"] = "multi-GB"):
        super(MultiModalModel_GB, self).__init__()
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
class TFN(nn.Module):
    def __init__(self, n_classes : int, hidden_dim : int, args_video : Dict, args_0D : Dict):
        super(TFN, self).__init__()
        self.n_classes = n_classes
        self.args_video = args_video
        self.args_0D = args_0D
        
        # Modality Embedding SubNetwork
        self.network_video = ViViTEncoder(**args_video)
        self.network_0D = ConvLSTMEncoder(**args_0D)
        
        self.network_0D_dims = self.network_0D.lstm_dim * 2
        self.network_video_dims = self.network_video.dim
        
        assert self.network_0D_dims == self.network_video_dims, "two encoder should be the same latent dims"
        
        self.encoder_dims = self.network_video_dims
        
        self.fusion_input_dims = (self.network_0D_dims + 1) * (self.network_video_dims + 1)

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_input_dims, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, x_vis : torch.Tensor, x_0D : torch.Tensor):
        h_vis = self.network_video(x_vis)
        h_0D = self.network_0D(x_0D)
          
        batch_size = h_vis.size()[0]
        
        _h_vis = torch.cat((Variable(torch.ones(batch_size, 1).float().to(x_vis.device), requires_grad=False), h_vis), dim=1)
        _h_0D = torch.cat((Variable(torch.ones(batch_size, 1).float().to(x_0D.device), requires_grad=False), h_0D), dim=1)
        
        fusion_tensor = torch.bmm(_h_vis.unsqueeze(2), _h_0D.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(batch_size, -1)
        
        out = self.classifier(fusion_tensor)
        
        return out
    
    def encode(self, x_vis : torch.Tensor, x_0D : torch.Tensor):
        with torch.no_grad():
            h_vis = self.network_video(x_vis)
            h_0D = self.network_0D(x_0D)
            
            batch_size = h_vis.size()[0]
            
            _h_vis = torch.cat((Variable(torch.ones(batch_size, 1).float().to(x_vis.device), requires_grad=False), h_vis), dim=1)
            _h_0D = torch.cat((Variable(torch.ones(batch_size, 1).float().to(x_0D.device), requires_grad=False), h_0D), dim=1)
        
            fusion_tensor = torch.bmm(_h_vis.unsqueeze(2), _h_0D.unsqueeze(1))
            fusion_tensor = fusion_tensor.view(batch_size, -1)

        return (fusion_tensor, h_vis, h_0D)

    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = False, print_summary : bool = True, show_parent_layers : bool = False):
        sample_video = torch.zeros((8,  self.args_video["in_channels"], self.args_video["n_frames"], self.args_video["image_size"], self.args_video["image_size"]), device = device)
        sample_0D = torch.zeros((8, self.args_0D["seq_len"], self.args_0D["col_dim"]), device = device)
        return summary(self, sample_video, sample_0D, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers)


# In this code, we also use Gradient Blending Method 
class TFN_GB(nn.Module):
    def __init__(self, n_classes : int, hidden_dim : int, args_video : Dict, args_0D : Dict):
        super(TFN_GB, self).__init__()
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
        
        assert self.network_0D_dims == self.network_video_dims, "two encoder should be the same latent dims"
        self.encoder_dims = self.network_video_dims
        self.fusion_input_dims = (self.network_0D_dims + 1) * (self.network_video_dims + 1)

        # Tensor Fusion Layer as classifier
        self.dropout = nn.Dropout(0)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_input_dims, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes)
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
        
        _h_vis = torch.cat((Variable(torch.ones(batch_size, 1).float().to(x_vis.device), requires_grad=False), h_vis), dim=1)
        _h_0D = torch.cat((Variable(torch.ones(batch_size, 1).float().to(x_0D.device), requires_grad=False), h_0D), dim=1)
        
        fusion_tensor = torch.bmm(_h_vis.unsqueeze(2), _h_0D.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(batch_size, -1)
        
        out = self.classifier(self.dropout(fusion_tensor))
        
        return (out, out_vis, out_0D)
    
    def encode(self, x_vis : torch.Tensor, x_0D : torch.Tensor):
        with torch.no_grad():
            latent_vis = self.embedd_subnet['network_video'].encode(x_vis)
            latent_0D = self.embedd_subnet['network_0D'].encode(x_0D)
            
            h_vis = latent_vis
            h_0D = latent_0D
            
            batch_size = h_vis.size()[0]

            _h_vis = torch.cat((Variable(torch.ones(batch_size, 1).float().to(x_vis.device), requires_grad=False), h_vis), dim=1)
            _h_0D = torch.cat((Variable(torch.ones(batch_size, 1).float().to(x_0D.device), requires_grad=False), h_0D), dim=1)
        
            fusion_tensor = torch.bmm(_h_vis.unsqueeze(2), _h_0D.unsqueeze(1))
            fusion_tensor = fusion_tensor.view(batch_size, -1)

        return (fusion_tensor, latent_vis, latent_0D)

    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = False, print_summary : bool = True, show_parent_layers : bool = False):
        sample_video = torch.zeros((8,  self.args_video["in_channels"], self.args_video["n_frames"], self.args_video["image_size"], self.args_video["image_size"]), device = device)
        sample_0D = torch.zeros((8, self.args_0D["seq_len"], self.args_0D["col_dim"]), device = device)
        return summary(self, sample_video, sample_0D, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers)