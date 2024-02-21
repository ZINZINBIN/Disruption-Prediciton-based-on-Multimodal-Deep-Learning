import hiddenlayer as hl
from torchviz import make_dot
from pytorch_model_summary import summary

from src.config import Config
import argparse
import torch
from typing import Dict

# Vision Network
from src.models.ViViT import ViViT
from src.models.R2Plus1D import R2Plus1DClassifier
from src.models.resnet import Bottleneck3D
from src.models.slowfast import SlowFast

# 0D Network
from src.models.transformer import Transformer
from src.models.CnnLSTM import CnnLSTM
from src.models.MLSTM_FCN import MLSTM_FCN

import warnings

# remove warning
warnings.filterwarnings("ignore")

config = Config()

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="hyperparameter tuning process for disruption prediction")
    
    # package
    parser.add_argument("--package", type = str, default = 'torchviz', choices=['torchviz', 'hiddenlayers'])
    
    # tag and result directory
    parser.add_argument("--model", type = str, default = 'Transformer', choices=['ViViT', 'SlowFast', 'R2Plus1D', 'Transformer','CnnLSTM','MLSTM_FCN'])

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)

    # data input shape 
    parser.add_argument("--image_size", type = int, default = 128)
    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--seq_len", type = int, default = 21)
       
    args = vars(parser.parse_args())

    return args

# argument parsing
args = parsing()

# hl.build_graph(model, input)

def load_model(model_argument:Dict):
    if args['model'] == 'ViViT':
        model = ViViT(
            image_size = args['image_size'],
            patch_size = model_argument['patch_size'],
            n_classes = 2,
            n_frames = args['seq_len'],
            dim = model_argument['dim'],
            depth = model_argument['depth'],
            n_heads = model_argument['n_heads'],
            pool = "mean",
            in_channels = 3,
            d_head = model_argument['d_head'],
            dropout = model_argument['dropout'],
            embedd_dropout=model_argument['embedd_dropout'],
            scale_dim = model_argument['scale_dim'],
            alpha = model_argument['alpha']
        )
        
    elif args['model'] == 'SlowFast':
                
        model = SlowFast(
            input_shape = (3, args['seq_len'] - 1, args['image_size'], args['image_size']),
            block = Bottleneck3D,
            layers = [1,model_argument['n_layer'],model_argument['n_layer'],1],
            alpha = model_argument['tau_alpha'],
            tau_fast = model_argument['tau_fast'],
            num_classes = 2,
            alpha_elu = model_argument['alpha'],
        )
        
    elif args['model'] == 'R2Plus1D':
        model = R2Plus1DClassifier(
            input_size  = (3, args['seq_len'], args['image_size'], args['image_size']),
            num_classes = 2, 
            layer_sizes = [1,model_argument['n_layer'],model_argument['n_layer'],1],
            pretrained = False, 
            alpha = model_argument['alpha']
        )
        
    elif args['model'] == 'Transformer':
        model = Transformer(
            n_features=len(config.input_features),
            feature_dims = model_argument['feature_dims'],
            max_len = args['seq_len'],
            n_layers = model_argument['n_layers'],
            n_heads = model_argument['n_heads'],
            dim_feedforward=model_argument['dim_feedforward'],
            dropout = model_argument['dropout'],
            cls_dims = model_argument['cls_dims'],
            n_classes = 2
        )
        
    elif args['model'] == 'CnnLSTM':
        model = CnnLSTM(
            seq_len = args['seq_len'],
            n_features=len(config.input_features),
            conv_dim = model_argument['conv_dim'],
            conv_kernel = model_argument['conv_kernel'],
            conv_stride=model_argument['conv_stride'],
            conv_padding=model_argument['conv_padding'],
            lstm_dim=model_argument['lstm_dim'],
            n_layers=model_argument['lstm_layers'],
            bidirectional=model_argument['bidirectional'],
            n_classes=2
        )
    
    elif args['model'] == 'MLSTM_FCN':
        model = MLSTM_FCN(
            n_features = len(config.input_features),
            fcn_dim = model_argument['fcn_dim'],
            kernel_size = model_argument['conv_kernel'],
            stride = model_argument['conv_stride'],
            seq_len = args['seq_len'],
            lstm_dim = model_argument['lstm_dim'],
            lstm_n_layers=model_argument['lstm_layers'],
            lstm_bidirectional=model_argument['bidirectional'],
            lstm_dropout=model_argument['lstm_dropout'],
            reduction = model_argument['reduction'],
            alpha = model_argument['alpha'],
            n_classes = 2
        )
       
    return model

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

# torch device state
print("############### device setup ###################")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# device allocation
if (torch.cuda.device_count() >= 1):
    device = "cuda:" + str(args["gpu_num"])
else:
    device = 'cpu'
    
if __name__ == "__main__":
    
    if args['model'] == 'ViViT':
        model_argument = {
            'patch_size':16,
            'dim':128,
            'depth':2,
            'n_heads':4,
            'd_head':64,
            'scale_dim':8,
            'dropout':0.1,
            'embedd_dropout':0.1,
            'alpha':1.0
        }
        
    elif args['model'] == 'SlowFast':
        model_argument = {
            'n_layer':2,
            'tau_alpha':4,
            'tau_fast':1,
            'alpha':1.0
        }  
    elif args['model'] == 'R2Plus1D':
        model_argument = {
            'n_layer':2,
            'alpha':1.0
        }
    elif args['model'] == 'Transformer':
        model_argument = {
            'feature_dims':128,
            'n_layers':4,
            'n_heads':8,
            'dim_feedforward':1024,
            'dropout':0.1,
            'cls_dims':128,
        }
        
    elif args['model'] == 'CnnLSTM':
        model_argument = {
            'conv_dim':64,
            'conv_kernel':3,
            'conv_stride':1,
            'conv_padding':1,
            'lstm_dim':128,
            'lstm_layers':4,
            'bidirectional':True
        }
    
    elif args['model'] == 'MLSTM_FCN':
        model_argument = {
            'fcn_dim':128,
            'conv_kernel':3,
            'conv_stride':1,
            'lstm_dim':128,
            'lstm_dropout':0.1,
            'lstm_layers':4,
            'bidirectional':True,
            'reduction':16,
            'alpha':1.0
        }
    
    if args["model"] in ["ViViT", "SlowFast", "R2Plus1D"]:
        
        if args["model"] == "SlowFast":
            input_shape = (1,3, args['seq_len'] - 1, args['image_size'],args['image_size'])
        else:
            input_shape = (1,3, args['seq_len'], args['image_size'],args['image_size'])
                    
    elif args["model"] in ["Transformer", "CnnLSTM", "MLSTM_FCN"]:
        input_shape = (1,args['seq_len'],len(config.input_features))
    
    model = load_model(model_argument)
    model.eval()
    model.to(device)
    data = torch.zeros(input_shape).to(device)
    
    with open("./results/architecture_{}.txt".format(args["model"]), 'w') as f:
        txt = summary(model, data, show_hierarchical=False, show_input = True, print_summary=True, max_depth=2)
        f.write(txt)
    
    if args['package'] == "torchviz":
        make_dot(model(data).mean(), params=dict(model.named_parameters()), show_attrs=False, show_saved=True).render( "./results/architecture_{}".format(args["model"]), format = "png")
    elif args['package'] == "hiddenlayers":
        im = hl.build_graph(model, data)
        im.save(path = "./results/architecture_{}".format(args["model"]), format = "png")
    