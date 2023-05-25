import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.dataset import DatasetFor0D
from torch.utils.data import DataLoader, RandomSampler
from src.utils.sampler import ImbalancedDatasetSampler
from src.utils.utility import preparing_0D_dataset, plot_learning_curve, generate_prob_curve_from_0D, seed_everything
from src.visualization.visualize_latent_space import visualize_2D_latent_space, visualize_3D_latent_space
from src.visualization.visualize_application import generate_real_time_experiment_0D
from src.train import train, train_DRW
from src.evaluate import evaluate
from src.loss import FocalLoss, LDAMLoss, CELoss
from src.models.transformer import Transformer
from src.models.CnnLSTM import CnnLSTM
from src.models.MLSTM_FCN import MLSTM_FCN
from src.feature_importance import compute_permute_feature_importance
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
    parser.add_argument("--model", type = str, default = 'Transformer', choices=['Transformer', 'CnnLSTM', 'MLSTM_FCN'])
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
    parser.add_argument("--lr", type = float, default = 2e-5)
    parser.add_argument("--use_scheduler", type = bool, default = True)
    parser.add_argument("--step_size", type = int, default = 4)
    parser.add_argument("--gamma", type = float, default = 0.95)
    
    # early stopping
    parser.add_argument('--early_stopping', type = bool, default = True)
    parser.add_argument("--early_stopping_patience", type = int, default = 64)
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
    parser.add_argument("--dropout", type = float, default = 0.25)
    parser.add_argument("--feature_dims", type = int, default = 128)
    parser.add_argument("--n_layers", type = int, default = 4)
    parser.add_argument("--n_heads", type = int, default = 8)
    parser.add_argument("--dim_feedforward", type = int, default = 512)
    parser.add_argument("--cls_dims", type = int, default = 128)
    
    # model setup : cnn lstm
    parser.add_argument("--conv_dim", type = int, default = 32)
    parser.add_argument("--conv_kernel", type = int, default = 3)
    parser.add_argument("--conv_stride", type = int, default = 1)
    parser.add_argument("--conv_padding", type = int, default = 1)
    parser.add_argument("--lstm_dim", type = int, default = 64)
    parser.add_argument("--lstm_layers", type = int, default = 2)
    parser.add_argument("--bidirectional", type = bool, default = True)
    
    # model setup : MLSTM_FCN
    parser.add_argument("--fcn_dim", type = int, default = 64)
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
        
    # dataset setup
    ts_train, ts_valid, ts_test, ts_scaler = preparing_0D_dataset("./dataset/KSTAR_Disruption_ts_data_extend.csv", ts_cols = ts_cols, scaler = 'Robust')
    kstar_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List.csv', encoding = "euc-kr")

    train_data = DatasetFor0D(ts_train, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 4 * 1 / 210, scaler = ts_scaler)
    valid_data = DatasetFor0D(ts_valid, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 4 * 1 / 210, scaler = ts_scaler)
    test_data = DatasetFor0D(ts_test, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 4 * 1 / 210, scaler = ts_scaler)
    
    print("================= Dataset information =================")
    print("train data : {}, disrupt : {}, non-disrupt : {}".format(train_data.__len__(), train_data.n_disrupt, train_data.n_normal))
    print("valid data : {}, disrupt : {}, non-disrupt : {}".format(valid_data.__len__(), valid_data.n_disrupt, valid_data.n_normal))
    print("test data : {}, disrupt : {}, non-disrupt : {}".format(test_data.__len__(), test_data.n_disrupt, test_data.n_normal))
    
    # label distribution for LDAM / Focal Loss
    train_data.get_num_per_cls()
    cls_num_list = train_data.get_cls_num_list()

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
    
    print("\n==================== model summary ====================\n")
    model.summary()
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
        
    # Re-sampling
    if args["use_sampling"]:
        train_sampler = ImbalancedDatasetSampler(train_data)
        valid_sampler = RandomSampler(valid_data)
        test_sampler = RandomSampler(test_data)

    else:
        train_sampler = RandomSampler(train_data)
        valid_sampler = RandomSampler(valid_data)
        test_sampler = RandomSampler(test_data)
    
    train_loader = DataLoader(train_data, batch_size = args['batch_size'], sampler=train_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    valid_loader = DataLoader(valid_data, batch_size = args['batch_size'], sampler=valid_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    test_loader = DataLoader(test_data, batch_size = args['batch_size'], sampler=test_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])

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

    # training process
    print("\n======================= training process =======================\n")
    if args['use_DRW']:
        train_loss,  train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train_DRW(
            train_loader,
            valid_loader,
            model,
            optimizer,
            loss_fn,
            device,
            args['num_epoch'],
            args['verbose'],
            save_best_dir = save_best_dir,
            save_last_dir = save_last_dir,
            exp_dir = exp_dir,
            max_norm_grad = 1.0,
            betas = betas,
            cls_num_list = cls_num_list,
            model_type = "single",
            test_for_check_per_epoch=test_loader,
            is_early_stopping = args['early_stopping'],
            early_stopping_verbose = args['early_stopping_verbose'],
            early_stopping_patience = args['early_stopping_patience'],
            early_stopping_delta = args['early_stopping_delta']
        )
        
    else:
        train_loss,  train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train(
            train_loader,
            valid_loader,
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            args['num_epoch'],
            args['verbose'],
            save_best_dir = save_best_dir,
            save_last_dir = save_last_dir,
            exp_dir = exp_dir,
            max_norm_grad = 1.0,
            model_type = "single",
            test_for_check_per_epoch=test_loader,
            is_early_stopping = args['early_stopping'],
            early_stopping_verbose = args['early_stopping_verbose'],
            early_stopping_patience = args['early_stopping_patience'],
            early_stopping_delta = args['early_stopping_delta']
        )
    
    # plot the learning curve
    save_learning_curve = os.path.join(save_dir, "{}_lr_curve.png".format(tag))
    plot_learning_curve(train_loss, valid_loss, train_f1, valid_f1, figsize = (12,6), save_dir = save_learning_curve)
    
    # evaluation process
    print("\n====================== evaluation process ======================\n")
    model.load_state_dict(torch.load(save_best_dir))
    
    save_conf = os.path.join(save_dir, "{}_test_confusion.png".format(tag))
    save_txt = os.path.join(save_dir, "{}_test_eval.txt".format(tag))
    
    test_loss, test_acc, test_f1 = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
        save_conf = save_conf,
        save_txt = save_txt
    )
    
    # compute the feature importance of the variables
    print("\n====================== Feature Importance ======================\n")
    compute_permute_feature_importance(
        model,
        test_loader,
        ts_cols,
        loss_fn,
        device,
        'single',
        'loss',
        os.path.join(save_dir, "{}_feature_importance.png".format(tag))
    )
    
    # Additional analyzation
    print("\n====================== Visualization process ======================\n")
    
    try:
        visualize_2D_latent_space(
            model, 
            train_loader,
            device,
            os.path.join(save_dir, "{}_2D_latent_train.png".format(tag))
        )
        
        visualize_2D_latent_space(
            model, 
            test_loader,
            device,
            os.path.join(save_dir, "{}_2D_latent_test.png".format(tag))
        )
        
    except:
        print("{} : visualize 2D latent space doesn't work due to stability error".format(tag))
    
    try:
        visualize_3D_latent_space(
            model, 
            train_loader,
            device,
            os.path.join(save_dir, "{}_3D_latent_train.png".format(tag))
        )
        
        visualize_3D_latent_space(
            model, 
            test_loader,
            device,
            os.path.join(save_dir, "{}_3D_latent_test.png".format(tag))
        )
    except:
        print("{} : visualize 3D latent space doesn't work due to stability error".format(tag))
    
    # plot probability curve
    test_shot_num = args['test_shot_num']

    print("\n====================== Probability curve generation process ======================\n")
    generate_prob_curve_from_0D(
        model, 
        device = device, 
        save_dir = os.path.join(save_dir, "{}_probs_curve_{}.png".format(tag, test_shot_num)),
        ts_data_dir = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
        ts_cols = ts_cols,
        shot_list_dir = './dataset/KSTAR_Disruption_Shot_List_extend.csv',
        shot_num = test_shot_num,
        seq_len = args['seq_len'],
        dist = args['dist'],
        dt = 4 / 210,
        scaler = ts_scaler
    )