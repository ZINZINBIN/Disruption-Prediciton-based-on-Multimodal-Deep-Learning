import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from src.models.resnet import *
from pytorch_model_summary import summary

torch.backends.cudnn.benchmark = True

# using slow-fast model : https://github.com/mbiparva/slowfast-networks-pytorch
class SlowNet(ResNet3D):
    def __init__(self, blocks, layers, **kwargs):
        super(SlowNet, self).__init__(blocks, layers, **kwargs)
        self.init_params()

    def forward(self, x : Tuple[torch.Tensor, List[torch.Tensor]]):
        x, laterals = x
        x = self.layer0(x)

        #print("after layer 0 x.size : ", x.size())
        x = torch.cat([x, laterals[0]], dim = 1)
        x = self.layer1(x)

        #print("after layer 1 x.size : ", x.size())
        x = torch.cat([x, laterals[1]], dim = 1)
        x = self.layer2(x)

        x = torch.cat([x, laterals[2]], dim = 1)
        x = self.layer3(x)

        x = torch.cat([x, laterals[3]], dim = 1)
        x = self.layer4(x)

        x = F.adaptive_avg_pool3d(x, 1)
        x = x.view(-1, x.size(1))

        return x

def resnet50_s(block = Bottleneck3D, layers = [3,4,6,3], **kwargs):
    model = SlowNet(block, layers, **kwargs)
    return model

class FastNet(ResNet3D):
    def __init__(self, blocks, layers, **kwargs):
        super(FastNet, self).__init__(blocks, layers, **kwargs)
        alpha = kwargs["alpha"]
        kernel_size = (alpha+2,1,1)
        stride = (alpha,1,1)
        padding = (1,0,0)
        
        m = 16

        stride_maxpool = (alpha, 1, 1)
        kernel_maxpool = (alpha + 2, 1, 1)

        self.l_maxpool = nn.Conv3d(m//self.alpha, m//self.alpha,
                                   kernel_size=kernel_maxpool, stride=stride_maxpool, bias=False, padding=padding)
        self.l_layer1 = nn.Conv3d(4*m//self.alpha, 4*m//self.alpha,
                                  kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.l_layer2 = nn.Conv3d(8*m//self.alpha, 8*m//self.alpha,
                                  kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.l_layer3 = nn.Conv3d(16*m//self.alpha, 16*m//self.alpha,
                                  kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.init_params()

    def forward(self, x : torch.Tensor):
        laterals = []

        x = self.layer0(x)
        laterals.append(self.l_maxpool(x))

        x = self.layer1(x)
        laterals.append(self.l_layer1(x))

        x = self.layer2(x)
        laterals.append(self.l_layer2(x))

        x = self.layer3(x)
        laterals.append(self.l_layer3(x))

        x = self.layer4(x)

        x = F.adaptive_avg_pool3d(x, 1)
        x = x.view(-1, x.size(1))

        return x, laterals

def resnet50_f(block = Bottleneck3D, layers = [3,4,6,3], **kwargs):
    model = FastNet(block, layers, **kwargs)
    return model

class SlowFastEncoder(nn.Module):
    def __init__(
            self, 
            input_shape : Tuple[int,int,int,int] = (3, 8, 112, 112),
            block : Optional[Bottleneck3D] = Bottleneck3D,
            layers : List[int] = [3,4,6,3],
            alpha : int = 4,
            tau_fast : int = 1,
        ):
        super(SlowFastEncoder, self).__init__()
        self.input_shape = input_shape
        self.seq_len = input_shape[1]
        self.in_channels = input_shape[0]
        self.alpha = alpha
        self.tau_fast = tau_fast

        self.slownet = resnet50_s(block = block, layers = layers, alpha = alpha, in_channels = self.in_channels, slow = 1, base_bn_splits = None)
        self.fastnet = resnet50_f(block = block, layers = layers, alpha = alpha, in_channels = self.in_channels, slow = 0, base_bn_splits = None)
        # self.slownet = SlowNet(blocks = block, layers = layers, alpha = alpha, in_channels = self.in_channels, slow = 1, base_bn_splits = None)
        # self.fastnet = FastNet(blocks = block, layers = layers, alpha = alpha, in_channels = self.in_channels, slow = 0, base_bn_splits = None)

    def split_slow_fast(self, x : torch.Tensor):
    
        tau_fast = self.tau_fast
        tau_slow = tau_fast * self.alpha

        x_slow = x[:,:,::tau_slow, :, :]
        x_fast = x[:,:,::tau_fast, :, :]

        return (x_slow, x_fast)

    def forward(self, x : torch.Tensor):
        x_slow, x_fast = self.split_slow_fast(x)
        x_fast, laterals = self.fastnet(x_fast)

        x_slow = self.slownet((x_slow, laterals))
        x = torch.cat([x_slow, x_fast], dim = 1)
        return x
    
    def show_CAM(self):
        pass

    def show_Grad_CAM(self):
        pass    
    
    def get_output_shape(self):
        input_shape = (1, *(self.input_shape))
        sample = torch.zeros(input_shape)
        sample_output = self.forward(sample)
        return sample_output.size()

class SlowFastClassifier(nn.Module):
    def __init__(
        self, 
        input_dim : int,
        num_classes : int = 2,
        alpha : float = 1.0,
    ):
        super(SlowFastClassifier, self).__init__()
        self.input_dim = input_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ELU(alpha),
            nn.Linear(input_dim // 2, num_classes)
        )

    def forward(self, x : torch.Tensor):
        x = self.classifier(x)
        return x
    
class SlowFast(nn.Module):
    def __init__(
        self,
        input_shape : Tuple[int,int,int,int] = (3, 8, 112, 112),
        block : Optional[Bottleneck3D] = Bottleneck3D,
        layers : List[int] = [3,4,6,3],
        alpha : int = 4,
        tau_fast : int = 1,
        num_classes : int = 2,
        alpha_elu : float = 1.0,
    ):
        super(SlowFast, self).__init__()
        
        self.input_shape = input_shape
        self.encoder = SlowFastEncoder(input_shape, block, layers, alpha, tau_fast)
        cls_input_dim = self.encoder.get_output_shape()[-1]
        self.classifier = SlowFastClassifier(cls_input_dim, num_classes, alpha_elu)

    def encode(self, x : torch.Tensor):
        with torch.no_grad():
            x = self.encoder.forward(x)
            batch = x.size()[0]
            x = x.view(batch, -1)
        return x
            
    def forward(self, x : torch.Tensor):
        x = self.encoder.forward(x)
        x = self.classifier.forward(x)
        return x

    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = True, print_summary : bool = False, show_parent_layers : bool = True):
        input_shape = (8, *(self.input_shape))
        sample = torch.zeros(input_shape, device = device)
        print(summary(self, sample, max_depth = 3, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers))