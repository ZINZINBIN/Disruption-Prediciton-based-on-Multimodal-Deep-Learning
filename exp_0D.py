import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.CustomDataset import DatasetFor0D
from torch.utils.data import DataLoader
from src.utils.utility import preparing_0D_dataset
from src.evaluate import evaluate_detail
from src.models.ts_transformer import TStransformer
from src.feature_importance import compute_permute_feature_importance
from src.loss import FocalLoss

# columns for use
ts_cols = ['\\q95', '\\ipmhd', '\\kappa', '\\tritop', '\\tribot','\\betap','\\betan','\\li', '\\WTOT_DLM03']

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Experiment for ViViT model")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "Transformer")
    parser.add_argument("--save_dir", type = str, default = "./results")

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)

    # common argument
    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 1024)
    parser.add_argument("--seq_len", type = int, default = 21)
    parser.add_argument("--dist", type = int, default = 3)
    parser.add_argument("--num_workers", type = int, default = 8)
    parser.add_argument("--pin_memory", type = bool, default = False)

    # model setup
    parser.add_argument("--alpha", type = float, default = 0.01)
    parser.add_argument("--dropout", type = float, default = 0.25)
    parser.add_argument("--feature_dims", type = int, default = 128)
    parser.add_argument("--n_layers", type = int, default = 8)
    parser.add_argument("--n_heads", type = int, default = 8)
    parser.add_argument("--dim_feedforward", type = int, default = 512)
    parser.add_argument("--cls_dims", type = int, default = 128)
    
    args = vars(parser.parse_args())

    return args

# torch device state
print("############### device setup ###################")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":

    args = parsing()
    
    # save directory
    save_dir = args['save_dir']
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    tag = "{}_clip_{}_dist_{}".format(args["tag"], args["seq_len"], args["dist"])
    save_best_dir = "./weights/{}_best.pt".format(tag)
    save_last_dir = "./weights/{}_last.pt".format(tag)
 
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'

    # dataset setup
    ts_train, ts_valid, ts_test, ts_scaler = preparing_0D_dataset("./dataset/KSTAR_Disruption_ts_data_extend.csv", ts_cols = ts_cols, scaler = 'Robust')
    kstar_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List.csv', encoding = "euc-kr")

    train_data = DatasetFor0D(ts_train, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 4 * 1 / 210)
    valid_data = DatasetFor0D(ts_valid, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 4 * 1 / 210)
    test_data = DatasetFor0D(ts_test, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 4 * 1 / 210)
    
    print("train data : {}, disrupt : {}, non-disrupt : {}".format(train_data.__len__(), train_data.n_disrupt, train_data.n_normal))
    print("valid data : {}, disrupt : {}, non-disrupt : {}".format(valid_data.__len__(), valid_data.n_disrupt, valid_data.n_normal))
    print("test data : {}, disrupt : {}, non-disrupt : {}".format(test_data.__len__(), test_data.n_disrupt, test_data.n_normal))
    
    # define model
    model = TStransformer(
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
    
    model.summary()
    model.to(device)
    
    train_loader = DataLoader(train_data, batch_size = args['batch_size'], sampler=None, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    valid_loader = DataLoader(valid_data, batch_size = args['batch_size'], sampler=None, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    test_loader = DataLoader(test_data, batch_size = args['batch_size'], sampler=None, num_workers = args["num_workers"], pin_memory=args["pin_memory"])

    model.load_state_dict(torch.load(save_best_dir))
    
    # loss definition
    train_data.get_num_per_cls()
    cls_num_list = train_data.get_cls_num_list()
    
    per_cls_weights = 1.0 / np.array(cls_num_list)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    focal_gamma = 2.0
    loss_fn = FocalLoss(weight = per_cls_weights, gamma = focal_gamma)
    
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
    
    save_csv = os.path.join(save_dir, "{}_total_score.csv".format(tag))
    
    evaluate_detail(
        train_loader,
        valid_loader,
        test_loader,
        model,
        device,
        save_csv,
        tag,
        model_type = 'single'
    )