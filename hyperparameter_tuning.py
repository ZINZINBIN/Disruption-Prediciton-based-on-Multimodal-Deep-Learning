import torch
import os
import numpy as np
import pandas as pd
import argparse
import logging
from typing import Dict, Optional, Literal
from functools import partial
from copy import deepcopy

# Dataset and Dataloader
from torch.utils.data import DataLoader, RandomSampler
from src.dataset import DatasetForVideo, DatasetFor0D
from src.utils.sampler import ImbalancedDatasetSampler
from src.utils.utility import preparing_video_dataset, preparing_0D_dataset, seed_everything

# train and evaluation function
from src.hpo import train, train_DRW, evaluate

# Loss function
from src.loss import FocalLoss, LDAMLoss, CELoss

# Vision Network
from src.models.ViViT import ViViT
from src.models.R2Plus1D import R2Plus1DClassifier
from src.models.resnet import Bottleneck3D
from src.models.slowfast import SlowFast

# 0D Network
from src.models.transformer import Transformer
from src.models.CnnLSTM import CnnLSTM
from src.models.MLSTM_FCN import MLSTM_FCN

# hyperparameter tuning library
import ray
from ray import tune, air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# ray-tune air : https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

# hyperopt : bayesian method searching algorithm
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
import warnings

# remove warning
warnings.filterwarnings("ignore")

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="hyperparameter tuning process for disruption prediction")
    
    # random seed
    parser.add_argument("--random_seed", type = int, default = 42)
    
    # tag and result directory
    parser.add_argument("--model", type = str, default = 'Transformer', choices=['ViViT', 'SlowFast', 'R2Plus1D', 'Transformer','CnnLSTM','MLSTM_FCN'])
    parser.add_argument("--tag", type = str, default = "Transformer")

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    parser.add_argument("--use_multi_gpu", type = bool, default = False)

    # data input shape 
    parser.add_argument("--image_size", type = int, default = 128)
    
    # num samples
    parser.add_argument("--num_samples", type = int, default = 16)

    # common argument
    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--num_epoch", type = int, default = 32)
    parser.add_argument("--seq_len", type = int, default = 21)
    parser.add_argument("--dist", type = int, default = 3)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = False)
    
    # detail setting for training process
    # data augmentation : conventional
    parser.add_argument("--bright_val", type = int, default = 10)
    parser.add_argument("--bright_p", type = float, default = 0.25)
    parser.add_argument("--contrast_min", type = float, default = 1)
    parser.add_argument("--contrast_max", type = float, default = 1.25)
    parser.add_argument("--contrast_p", type = float, default = 0.25)
    parser.add_argument("--blur_k", type = int, default = 5)
    parser.add_argument("--blur_p", type = float, default = 0.25)
    parser.add_argument("--flip_p", type = float, default = 0.25)
    parser.add_argument("--vertical_ratio", type = float, default = 0.1)
    parser.add_argument("--vertical_p", type = float, default = 0.25)
    parser.add_argument("--horizontal_ratio", type = float, default = 0.1)
    parser.add_argument("--horizontal_p", type = float, default = 0.25)
    
    # optimizer : SGD, RMSProps, Adam, AdamW
    parser.add_argument("--optimizer", type = str, default = "AdamW", choices=["SGD","RMSProps","Adam","AdamW"])
    
    # learning rate, step size and decay constant
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--use_scheduler", type = bool, default = True)
    parser.add_argument("--step_size", type = int, default = 4)
    parser.add_argument("--gamma", type = float, default = 0.95)
        
    # imbalanced dataset processing
    # Re-sampling
    parser.add_argument("--use_sampling", type = bool, default = False)
    
    # Re-weighting
    parser.add_argument("--use_weighting", type = bool, default = False)
    
    # Deffered Re-weighting
    parser.add_argument("--use_DRW", type = bool, default = True)
    parser.add_argument("--beta", type = float, default = 0.25)

    # loss type : CE, Focal, LDAM
    parser.add_argument("--loss_type", type = str, default = "Focal", choices = ['CE','Focal', 'LDAM'])
    
    # LDAM Loss parameter
    parser.add_argument("--max_m", type = float, default = 0.5)
    parser.add_argument("--s", type = float, default = 1.0)
    
    # Focal Loss parameter
    parser.add_argument("--focal_gamma", type = float, default = 2.0)
    
    # monitoring the training process
    parser.add_argument("--verbose", type = int, default = 16)
    
    args = vars(parser.parse_args())

    return args

# argument parsing
args = parsing()

ray.shutdown()  # Restart Ray defensively in case the ray connection is lost. 
ray.init(log_to_driver=True, ignore_reinit_error=True)

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

# seed initialize
seed_everything(args['random_seed'], False)
    
if args['model'] == 'SlowFast' and args['seq_len'] % 2 == 1:
    print("SlowFast : seq_len must be even number, seq_len-1 as input")
    args['seq_len'] -= 1

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

tag = "{}_clip_{}_dist_{}_{}_{}_hpo".format(args["tag"], args["seq_len"], args["dist"], loss_type, boost_type)

# directory for checkpoint
if not os.path.isdir("./hpo_checkpoint"):
    os.mkdir("./hpo_checkpoint")

checkpoint_dir = os.path.join("./hpo_checkpoint", tag)

print("HPO running process | model : {}".format(tag))

# To restore a checkpoint, use `session.get_checkpoint()`.
loaded_checkpoint = session.get_checkpoint()
if loaded_checkpoint:
    checkpoint_dir = loaded_checkpoint.as_directory()
        
# augmentation argument
augment_args = {
    "bright_val" : args['bright_val'],
    "bright_p" : args['bright_p'],
    "contrast_min" : args['contrast_min'],
    "contrast_max" : args['contrast_max'],
    "contrast_p" : args['contrast_p'],
    "blur_k" : args['blur_k'],
    "blur_p" : args['blur_p'],
    "flip_p" : args['flip_p'],
    "vertical_ratio" : args['vertical_ratio'],
    "vertical_p" : args['vertical_p'],
    "horizontal_ratio" : args['horizontal_ratio'],
    "horizontal_p" : args['horizontal_p']
}

# 0D data feature columns
ts_cols = [
    '\\q95', '\\ipmhd', '\\kappa', '\\tritop', '\\tribot',
    '\\betap','\\betan','\\li', '\\WTOT_DLM03','\\ne_inter01', 
    '\\TS_NE_CORE_AVG', '\\TS_TE_CORE_AVG'
]

# Dataset for vision network
if args["model"] in ["R2Plus1D", "SlowFast", "ViViT"]:
    root_dir = "./dataset/temp"
    shot_train, shot_valid, shot_test = preparing_video_dataset(root_dir)
    df_disrupt = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_extend.csv")
    
    train_data = DatasetForVideo(shot_train, df_disrupt, augmentation = True, augmentation_args=augment_args, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])
    valid_data = DatasetForVideo(shot_valid, df_disrupt, augmentation = False, augmentation_args=augment_args, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])
    test_data = DatasetForVideo(shot_test, df_disrupt, augmentation = False, augmentation_args=augment_args, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])

elif args["model"] in ["Transformer","CnnLSTM","MLSTM_FCN"]:
    ts_train, ts_valid, ts_test, ts_scaler = preparing_0D_dataset("./dataset/KSTAR_Disruption_ts_data_extend.csv", ts_cols = ts_cols, scaler = 'Robust')
    kstar_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List.csv', encoding = "euc-kr")
    train_data = DatasetFor0D(ts_train, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 4 * 1 / 210, scaler = ts_scaler)
    valid_data = DatasetFor0D(ts_valid, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 4 * 1 / 210, scaler = ts_scaler)
    test_data = DatasetFor0D(ts_test, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 4 * 1 / 210, scaler = ts_scaler)
    
# label distribution for LDAM / Focal Loss
train_data.get_num_per_cls()
cls_num_list = train_data.get_cls_num_list()

# Re-sampling
if args["use_sampling"]:
    train_sampler = ImbalancedDatasetSampler(train_data)
    valid_sampler = RandomSampler(valid_data)
    test_sampler = RandomSampler(test_data)

else:
    train_sampler = RandomSampler(train_data)
    valid_sampler = RandomSampler(valid_data)
    test_sampler = RandomSampler(test_data)

# Re-weighting
if args['use_weighting']:
    per_cls_weights = 1.0 / np.array(cls_num_list)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
else:
    per_cls_weights = np.array([1,1])
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
# loss
if args['loss_type'] == "CE":
    betas = [0, args['beta'], args['beta'] * 2, args['beta']*3]
    loss_fn = CELoss(weight = per_cls_weights)
elif args['loss_type'] == 'LDAM':
    max_m = args['max_m']
    s = args['s']
    betas = [0, args['beta'], args['beta'] * 2, args['beta']*3]
    loss_fn = LDAMLoss(cls_num_list, max_m = max_m, s = s, weight = per_cls_weights)
elif args['loss_type'] == 'Focal':
    betas = [0, args['beta'], args['beta'] * 2, args['beta']*3]
    focal_gamma = args['focal_gamma']
    loss_fn = FocalLoss(weight = per_cls_weights, gamma = focal_gamma)
else:
    betas = [0, args['beta'], args['beta'] * 2, args['beta']*3]
    loss_fn = CELoss(weight = per_cls_weights)
    
train_loader = DataLoader(train_data, batch_size = args['batch_size'], sampler=train_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
valid_loader = DataLoader(valid_data, batch_size = args['batch_size'], sampler=valid_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
test_loader = DataLoader(test_data, batch_size = args['batch_size'], sampler=test_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    
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
            input_shape = (3, args['seq_len'], args['image_size'], args['image_size']),
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
            n_features=len(ts_cols),
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
            n_features=len(ts_cols),
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
            n_features = len(ts_cols),
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

def train_for_hpo(
    config: Dict,
    checkpoint_dir : str,
    train_loader,
    valid_loader,
    ):
    
    # define model
    model = load_model(config)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:{}".format(args['gpu_num'])
        if torch.cuda.device_count() > 1 and args['use_multi_gpu']:
            model = torch.nn.DataParallel(model)

    model.to(device)

    # optimizer
    if args["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "RMSProps":
        optimizer = torch.optim.RMSprop(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    
    # scheduler
    if args["use_scheduler"] and not args["use_DRW"]:    
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args['step_size'], gamma=args['gamma'])
        
    elif args["use_DRW"]:
        scheduler = "DRW"
        
    else:
        scheduler = None
    
    # training process
    if args['use_DRW']:
        train_loss,  train_f1, valid_loss, valid_f1 = train_DRW(
            train_loader,
            valid_loader,
            model,
            optimizer,
            loss_fn,
            device,
            args['num_epoch'],
            max_norm_grad = 1.0,
            cls_num_list=cls_num_list,
            betas = betas,
            model_type = "single",
            checkpoint_dir=checkpoint_dir
        )
        
    else:
        train_loss, train_f1, valid_acc, valid_f1 = train(
            train_loader,
            valid_loader,
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            args['num_epoch'],
            max_norm_grad = 1.0,
            model_type = "single",
            checkpoint_dir=checkpoint_dir
        )
    
    return


if __name__ == "__main__":
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    num_samples = args['num_samples']
    cpus_per_trial = 1
    gpus_per_trial = 4
    
    # define model
    if args['model'] == 'ViViT':
        model_argument = {
            'patch_size':tune.choice([8, 16, 32]),
            'dim':tune.sample_from(lambda _: 2**np.random.randint(5,10)),
            'depth':tune.choice([2,4,6,8]),
            'n_heads':tune.choice([2,4,6,8]),
            'd_head':tune.sample_from(lambda _: 2**np.random.randint(4,8)),
            'scale_dim':tune.sample_from(lambda _: 2**np.random.randint(1,4)),
            'dropout':tune.loguniform(1e-2, 5e-1),
            'embedd_dropout':tune.loguniform(1e-2, 5e-1),
            'alpha':tune.loguniform(1e-1, 1)
        }
        
    elif args['model'] == 'SlowFast':
        model_argument = {
            'n_layer':tune.choice([1,2,3,4]),
            'tau_alpha':4,
            'tau_fast':tune.choice([1,2]),
            'alpha':tune.loguniform(1e-1, 1)
        }  
    elif args['model'] == 'R2Plus1D':
        model_argument = {
            'n_layer':tune.choice([1,2,3,4]),
            'alpha':tune.loguniform(1e-1, 1)
        }
    elif args['model'] == 'Transformer':
        model_argument = {
            'feature_dims':tune.sample_from(lambda _: 2**np.random.randint(6,9)),
            'n_layers':tune.choice([2,4,6,8]),
            'n_heads':tune.choice([2,4,8]),
            'dim_feedforward':tune.sample_from(lambda _: 2**np.random.randint(7,10)),
            'dropout':tune.loguniform(1e-2, 2e-1),
            'cls_dims':tune.sample_from(lambda _: 2**np.random.randint(6,8)),
        }
        
    elif args['model'] == 'CnnLSTM':
        model_argument = {
            'conv_dim':tune.sample_from(lambda _: 2**np.random.randint(5,7)),
            'conv_kernel':tune.choice([3,5,7]),
            'conv_stride':tune.choice([1,2]),
            'conv_padding':1,
            'lstm_dim':tune.sample_from(lambda _: 2**np.random.randint(5,7)),
            'lstm_layers':tune.choice([1,2,3,4]),
            'bidirectional':True
        }
    
    elif args['model'] == 'MLSTM_FCN':
        model_argument = {
            'fcn_dim':tune.sample_from(lambda _: 2**np.random.randint(5,7)),
            'conv_kernel':tune.choice([3,5,7]),
            'conv_stride':tune.choice([1,2]),
            'lstm_dim':tune.sample_from(lambda _: 2**np.random.randint(5,7)),
            'lstm_dropout':tune.loguniform(1e-2, 5e-1),
            'lstm_layers':tune.choice([1,2,3,4]),
            'bidirectional':True,
            'reduction':tune.choice([4,8,16]),
            'alpha':tune.loguniform(1e-1, 1)
        }
    
    tune_scheduler = ASHAScheduler(
        metric="f1_score",
        mode="max",
        max_t = args['num_epoch'],
        grace_period=1,
        reduction_factor=2
    )
    
    tune_reporter = CLIReporter(
        metric_columns=["loss", "f1_score", "training_iteration"],
        parameter_columns=[key for key in model_argument.keys()]
    )
    
    # the solution : https://stackoverflow.com/questions/69777578/the-actor-implicitfunc-is-too-large-error
    trainable = tune.with_parameters(
        train_for_hpo,
        checkpoint_dir = checkpoint_dir,
        train_loader = train_loader,
        valid_loader = valid_loader,
    )
        
    tune_result = tune.run(
        trainable,
        resources_per_trial={
            "cpu":cpus_per_trial, 
            "gpu": gpus_per_trial
        },
        local_dir = checkpoint_dir,
        config = model_argument,
        num_samples=num_samples,
        scheduler=tune_scheduler,
        progress_reporter=tune_reporter,
        checkpoint_at_end=False
    )
        
    best_trial = tune_result.get_best_trial("f1_score", "max", "last")
    print("Best trial final validation f1 score : {:.3f}".format(best_trial.last_result["f1_score"]))
    
    print("Best trial config: {}".format(best_trial.config))
    
    best_model = load_model(best_trial.config)
    best_model.to(device)
    
    best_checkpoint = best_trial.checkpoint
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint.dir_or_data, "checkpoint"))
    
    best_model.load_state_dict(model_state) 
    
    test_loss, test_f1, test_auc = evaluate(
        test_loader,
        best_model,
        loss_fn,
        device,
        0.5,
        'single'
    )
    
    print("Best trial test f1-score:{:.3f}, test AUC:{:.3f}".format(test_f1, test_auc))
    
    
    ray.shutdown()