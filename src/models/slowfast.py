import numpy as np
from torch import unsqueeze
import torch 
import torch.nn as nn
from typing import Tuple, List
from src.models.layer import *
from src.models.transformer import *
from src.models.resnet import *
from src.models.abstract import *
from pytorch_model_summary import summary

# using slow-fast model : https://github.com/mbiparva/slowfast-networks-pytorch
class SlowNet(ResNet3D):
    def __init__(self, blocks, layers, **kwargs):
        super(SlowNet, self).__init__(blocks, layers, **kwargs)
        self.init_params()

    def forward(self, x)->torch.Tensor:
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

        stride_maxpool = (alpha, 1, 1)
        kernel_maxpool = (alpha + 2, 1, 1)

        self.l_maxpool = nn.Conv3d(64//self.alpha, 64//self.alpha,
                                   kernel_size=kernel_maxpool, stride=stride_maxpool, bias=False, padding=padding)
        self.l_layer1 = nn.Conv3d(4*64//self.alpha, 4*64//self.alpha,
                                  kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.l_layer2 = nn.Conv3d(8*64//self.alpha, 8*64//self.alpha,
                                  kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.l_layer3 = nn.Conv3d(16*64//self.alpha, 16*64//self.alpha,
                                  kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.init_params()

    def forward(self, x : torch.Tensor)->torch.Tensor:
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

class SlowFastEncoder(AbstractEncoder):
    sample_batch = 2
    def __init__(
            self, 
            input_shape : Tuple[int,int,int,int] = (3, 8, 112, 112),
            block : Optional[Bottleneck3D] = Bottleneck3D,
            layers : List[int] = [3,4,6,3],
            alpha : int = 4,
            tau_fast : int = 1,
            device : Optional[str] = None
        ):

        self.input_shape = input_shape
        self.seq_len = input_shape[1]
        self.in_channels = input_shape[0]
        self.alpha = alpha
        self.tau_fast = tau_fast

        if device is None:
            self.device = next(self.slownet.parameters()).device
        else:
            self.device = device

        self.slownet = resnet50_s(block = block, layers = layers, alpha = alpha, in_channels = self.in_channels, slow = 1)
        self.fastnet = resnet50_f(block = block, layers = layers, alpha = alpha, in_channels = self.in_channels, slow = 0)

    def split_slow_fast(self, x : torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        tau_fast = self.tau_fast
        tau_slow = tau_fast * self.alpha

        x_slow = x[:,:,::tau_slow, :, :]
        x_fast = x[:,:,::tau_fast, :, :]

        return (x_slow, x_fast)

    def forward(self, x : torch.Tensor)->torch.Tensor:
        x_slow, x_fast = self.split_slow_fast(x)
        x_fast, laterals = self.fastnet(x_fast)

        x_slow = self.slownet((x_slow, laterals))
        x = torch.cat([x_slow, x_fast], dim = 1)
        return x
    
    def device_allocation(self, device: str)->None:
        self.slownet.to(device)
        self.fastnet.to(device)
        self.device = device
    
    def show_strucuture(self)->None:
        sample = torch.zeros((self.sample_batch, *(self.input_shape))).to(self.device)
        print(summary(self, sample, max_depth = None, show_parent_layers = True, show_input = True))

    def show_CAM(self):
        pass

    def show_Grad_CAM(self):
        pass    
    
    def get_output_shape(self):
        input_shape = (self.sample_batch, *(self.input_shape))
        sample = torch.zeros(input_shape).to(self.device)
        sample_output = self.forward(sample)
        return sample_output.size()

class SlowFastClassifier(AbstractClassifier):
    sample_batch = 2
    def __init__(
        self, 
        input_dim : int,
        mlp_hidden :int = 128,
        num_classes : int = 2,
        device : Optional[str] = None
    ):
        self.input_dim = input_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_classes)
        )

        if device is None:
            self.device = next(self.classifier.parameters()).device
        else:
            self.device = device

    def forward(self, x : torch.Tensor)->torch.Tensor:
        x = self.classifier(x)
        return x
    
    def show_strucuture(self)->None:
        sample = torch.zeros((self.sample_batch, self.input_dim)).to(self.device)
        print(summary(self, sample, max_depth = None, show_parent_layers = True, show_input = True))

    def device_allocation(self, device: str)->None:
        self.classifier.to(device)
        self.device = device