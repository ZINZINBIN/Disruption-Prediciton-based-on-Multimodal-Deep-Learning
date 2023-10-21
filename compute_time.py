import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.utils.utility import seed_everything
from src.models.transformer import Transformer
from src.models.CnnLSTM import CnnLSTM
from src.models.MLSTM_FCN import MLSTM_FCN
from src.models.ViViT import ViViT
from src.config import Config

# columns for use
# ip error : target - measure value
# ne_inter03 : nan value 
# ne_tci01 : since 2018, minus(shot : 21272)
# 2018년도 샷은 ne가 이상해서 쓰지 않는 것 추천 / 확인해볼 것
# shot 31356 -> abrupt disruption, 다른 진단이 필요하다, LM
# shot 31243 -> abrupt disruption, 
# shot 31676 -> 
# warning region : ~ 400ms? 정도로 확대해서 disruption prediction 해보자
# flag2 : no train, no valid (20%, 240)

# Heaing factor, EC, NBI
# shot list 확장 + prediction region(warning time ~ 400ms)
# abrupt disruption shot 

# input_shape : [Batch, sequence lenth, ]

'''
Te = [x1,x2,x3,x4,x5, .... , x50]
   = [0,0,0,0,...,100,120,...,100,0,0,0,]

Idea : core value or specific position (r = 1.8대비 r = 1.5 등), shot마다 다른지 확인
* edge : error가 높아서 좋지 않음(accuracy bad)
* lock mode : ikstar 참조, work/disruption/machinelearning/database/data, LM
* lock mode에 대한 numbering도 포함
* DB : 32, lock mode error
'''

config = Config()

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="training disruption prediction model with 0D data")
    
    # random seed
    parser.add_argument("--random_seed", type = int, default = 42)
    
    # tag and result directory
    parser.add_argument("--model", type = str, default = 'Transformer', choices=['Transformer', 'CnnLSTM', 'MLSTM_FCN', 'ViViT'])
    parser.add_argument("--tag", type = str, default = "Transformer")
    parser.add_argument("--save_dir", type = str, default = "./results")
    
    # test shot for disruption probability curve
    parser.add_argument("--test_shot_num", type = int, default = 21310)

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)

    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 256)
    parser.add_argument("--num_epoch", type = int, default = 128)
    parser.add_argument("--seq_len", type = int, default = 21)
    parser.add_argument("--dist", type = int, default = 3)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = True)
    
    # optimizer : SGD, RMSProps, Adam, AdamW
    parser.add_argument("--optimizer", type = str, default = "AdamW", choices=["SGD","RMSProps","Adam","AdamW"])
    
    # learning rate, step size and decay constant
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--use_scheduler", type = bool, default = True)
    parser.add_argument("--step_size", type = int, default = 4)
    parser.add_argument("--gamma", type = float, default = 0.995)
    
    # early stopping
    parser.add_argument('--early_stopping', type = bool, default = True)
    parser.add_argument("--early_stopping_patience", type = int, default = 32)
    parser.add_argument("--early_stopping_verbose", type = bool, default = True)
    parser.add_argument("--early_stopping_delta", type = float, default = 1e-3)

    # imbalanced dataset processing
    # Re-sampling
    parser.add_argument("--use_sampling", type = bool, default = False)
    
    # Re-weighting
    parser.add_argument("--use_weighting", type = bool, default = False)
    
    # Deffered Re-weighting
    parser.add_argument("--use_DRW", type = bool, default = False)
    parser.add_argument("--beta", type = float, default = 0.25)

    # loss type : CE, Focal, LDAM
    parser.add_argument("--loss_type", type = str, default = "Focal", choices = ['CE','Focal', 'LDAM'])
    
    # LDAM Loss parameter
    parser.add_argument("--max_m", type = float, default = 0.5)
    parser.add_argument("--s", type = float, default = 1.0)
    
    # Focal Loss parameter
    parser.add_argument("--focal_gamma", type = float, default = 2.0)
    
    # monitoring the training process
    parser.add_argument("--verbose", type = int, default = 4)
    
    # model setup : transformer
    parser.add_argument("--alpha", type = float, default = 0.01)
    parser.add_argument("--dropout", type = float, default = 0.1)
    parser.add_argument("--feature_dims", type = int, default = 128)
    parser.add_argument("--n_layers", type = int, default = 4)
    parser.add_argument("--n_heads", type = int, default = 8)
    parser.add_argument("--dim_feedforward", type = int, default = 1024)
    parser.add_argument("--cls_dims", type = int, default = 128)
    
    # model setup : cnn lstm
    parser.add_argument("--conv_dim", type = int, default = 64)
    parser.add_argument("--conv_kernel", type = int, default = 3)
    parser.add_argument("--conv_stride", type = int, default = 1)
    parser.add_argument("--conv_padding", type = int, default = 1)
    parser.add_argument("--lstm_dim", type = int, default = 128)
    parser.add_argument("--lstm_layers", type = int, default = 4)
    parser.add_argument("--bidirectional", type = bool, default = True)
    
    # model setup : MLSTM_FCN
    parser.add_argument("--fcn_dim", type = int, default = 128)
    parser.add_argument("--reduction", type = int, default = 16)
    
    args = vars(parser.parse_args())

    return args

# torch device state
print("================= device setup =================")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()


if __name__ == "__main__":

    args = parsing()
    
    # seed initialize
    seed_everything(args['random_seed'], False)
    
    # save directory
    save_dir = args['save_dir']
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    # tag : {model_name}_clip_{seq_len}_dist_{pred_len}_{Loss-type}_{Boosting-type}
    loss_type = args['loss_type']
    
    if args['use_sampling'] and not args['use_weighting'] and not args['use_DRW']:
        boost_type = "RS"
    elif args['use_sampling'] and args['use_weighting'] and not args['use_DRW']:
        boost_type = "RS_RW"
    elif args['use_sampling'] and not args['use_weighting'] and args['use_DRW']:
        boost_type = "RS_DRW"
    elif args['use_sampling'] and args['use_weighting'] and args['use_DRW']:
        boost_type = "RS_DRW"
    elif not args['use_sampling'] and args['use_weighting'] and not args['use_DRW']:
        boost_type = "RW"
    elif not args['use_sampling'] and not args['use_weighting'] and args['use_DRW']:
        boost_type = "DRW"
    elif not args['use_sampling'] and args['use_weighting'] and args['use_DRW']:
        boost_type = "DRW"
    elif not args['use_sampling'] and not args['use_weighting'] and not args['use_DRW']:
        boost_type = "Normal"
    
    tag = "{}_clip_{}_dist_{}_{}_{}_seed_{}".format(args["tag"], args["seq_len"], args["dist"], loss_type, boost_type, args['random_seed'])
    
    print("================= Running code =================")
    print("Setting : {}".format(tag))
    
    save_best_dir = "./weights/{}_best.pt".format(tag)
    save_last_dir = "./weights/{}_last.pt".format(tag)
    exp_dir = os.path.join("./runs/", "tensorboard_{}".format(tag))
    
    # input features
    ts_cols = config.input_features
 
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
   
    # define model
    if args['model'] == 'Transformer':
        
        model = Transformer(
            n_features=len(ts_cols),
            feature_dims = args['feature_dims'],
            max_len = args['seq_len'],
            n_layers = args['n_layers'],
            n_heads = args['n_heads'],
            dim_feedforward=args['dim_feedforward'],
            dropout = args['dropout'],
            cls_dims = args['cls_dims'],
            n_classes = 2
        )
        
    elif args['model'] == 'CnnLSTM':
        
        model = CnnLSTM(
            seq_len = args['seq_len'],
            n_features=len(ts_cols),
            conv_dim = args['conv_dim'],
            conv_kernel = args['conv_kernel'],
            conv_stride=args['conv_stride'],
            conv_padding=args['conv_padding'],
            lstm_dim=args['lstm_dim'],
            n_layers=args['lstm_layers'],
            bidirectional=args['bidirectional'],
            n_classes=2
        )
    
    elif args['model'] == 'MLSTM_FCN':
        model = MLSTM_FCN(
            n_features = len(ts_cols),
            fcn_dim = args['fcn_dim'],
            kernel_size = args['conv_kernel'],
            stride = args['conv_stride'],
            seq_len = args['seq_len'],
            lstm_dim = args['lstm_dim'],
            lstm_n_layers=args['lstm_layers'],
            lstm_bidirectional=args['bidirectional'],
            lstm_dropout=0.1,
            reduction = args['reduction'],
            alpha = args['alpha'],
            n_classes = 2
        )
        
    elif args['model'] == 'ViViT':
        model = ViViT(
            image_size = 128,
            patch_size = 16,
            n_classes = 2,
            n_frames = 21,
            dim = 128,
            depth = 2,
            n_heads = 8,
            pool = "mean",
            in_channels = 3,
            d_head = 64,
            dropout = 0.1,
            embedd_dropout=0.1,
            scale_dim = 8,
            alpha = 0.01
        )
    
    model.summary()
    model.to(device)

    from src.utils.utility import measure_computation_time
    if args['model'] != 'ViViT':
        t_avg, t_std, t_measures  = measure_computation_time(model, (1, 21, len(ts_cols)), n_samples = 16, device = device)
    else:
        t_avg, t_std, t_measures  = measure_computation_time(model, (1, 21, 3, 128, 128), n_samples = 16, device = device)
    
    print("t_avg : {:.3f}, t_std : {:.3f}".format(t_avg, t_std))