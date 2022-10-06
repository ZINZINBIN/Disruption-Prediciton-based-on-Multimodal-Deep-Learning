import torch
import argparse
import numpy as np
import pandas as pd
import gc
import os
from src.CustomDataset import DatasetFor0D
from src.models.ConvLSTM import ConvLSTM
from src.models.ts_transformer import TStransformer
from src.utils.sampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from typing import Dict
from src.train import train, train_DRW
from src.evaluate import evaluate
from src.loss import LDAMLoss, FocalLoss
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from src.visualization.visualize_latent_space import visualize_3D_latent_space

parser = argparse.ArgumentParser(description="experiment for ConvLSTM")
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--loss_type", type = str, default = 'LDAM')
args = vars(parser.parse_args())

df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index()

# nan interpolation
df.interpolate(method = 'linear', limit_direction = 'forward')

# columns for use
ts_cols = [
    '\\q95', '\\ipmhd', '\\kappa', 
    '\\tritop', '\\tribot','\\betap','\\betan',
    '\\li', '\\WTOT_DLM03'
]

# float type
for col in ts_cols:
    df[col] = df[col].astype(np.float32)

# train / valid / test data split
from sklearn.model_selection import train_test_split
shot_list = np.unique(df.shot.values)

shot_train, shot_test = train_test_split(shot_list, test_size = 0.2, random_state = 42)
shot_train, shot_valid = train_test_split(shot_train, test_size = 0.2, random_state = 42)

df_train = pd.DataFrame()
df_valid = pd.DataFrame()
df_test = pd.DataFrame()

for shot in shot_train:
    df_train = pd.concat([df_train, df[df.shot == shot]], axis = 0)

for shot in shot_valid:
    df_valid = pd.concat([df_valid, df[df.shot == shot]], axis = 0)

for shot in shot_test:
    df_test = pd.concat([df_test, df[df.shot == shot]], axis = 0)

scaler = RobustScaler()
df_train[ts_cols] = scaler.fit_transform(df_train[ts_cols].values)
df_valid[ts_cols] = scaler.transform(df_valid[ts_cols].values)
df_test[ts_cols] = scaler.transform(df_test[ts_cols].values)

# disruption info
kstar_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_extend.csv', encoding = "euc-kr")

# shot list
shot_list = np.unique(df.shot.values).tolist()

ts_train = df_train
ts_valid = df_valid
ts_test = df_test

DEFAULT_ARGS = {
    "batch_size":64,
    "lr" : 1e-3,
    "gamma" : 0.95,
    "gpu_num" : args['gpu_num'],
    "seq_len" : 21,
    "col_dim" : len(ts_cols), 
    "conv_dim" : 32, 
    "conv_kernel" : 3,
    "conv_stride" : 1, 
    "conv_padding" : 1,
    "lstm_dim" : 64, 
    "n_classes" : 2, 
    "mlp_dim" : 64,
    "num_workers" : 8,
    "pin_memory" : False,
    "use_sampler" : True,
    "num_epoch" : 64,
    "verbose" : 8,
    "save_best_dir" : "./weights/ConvLSTM_clip_21_dist_5_best.pt",
    "save_last_dir" : "./weights/ConvLSTM_clip_21_dist_5_last.pt",
    "save_result_dir" : "./results/train_valid_loss_acc_ConvLSTM_clip_21_dist_5.png",
    "save_txt" : "./results/test_ConvLSTM_clip_21_dist_5.txt",
    "save_conf" : "./results/test_ConvLSTM_clip_21_dist_5_confusion_matrix.png",
    "save_latent" : "./results/test_ConvLSTM_clip_21_dist_3_latent.png",
    "use_focal_loss" : True if args['loss_type'] == 'FOCAL' else False,
    "use_LDAM_loss" : True if args['loss_type'] == 'LDAM' else False,
    "use_weight" : False,
    "use_DRW" : False,
    "root" : "dur21_dis3",
}


# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

def scheduling(args : Dict, idx : int, loss_type : str):

    weight_path = os.path.join("./weights", "experiment_ConvLSTM_{}".format(loss_type)) 
    result_path = os.path.join("./results", "experiment_ConvLSTM_{}".format(loss_type))  
    
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if idx == 0:
        args['use_sampler'] = True
        args['use_weight'] = True
        args['use_DRW'] = False
        title = "RS_RW"
    
    else:
        args['use_sampler'] = True
        args['use_weight'] = False
        args['use_DRW'] = True
        title = "DRW_RS"

    save_best_dir = os.path.join(weight_path, args['root'] + "_{}.pt".format(title))
    save_last_dir = os.path.join(weight_path, args['root'] + "_{}.pt".format(title))
    save_result_dir = os.path.join(result_path, args['root'] + "_loss_curve_{}.png".format(title))
    save_txt = os.path.join(result_path, args['root'] + "_eval_{}.txt".format(title))
    save_conf = os.path.join(result_path, args['root'] + "_confusion_{}.png".format(title))
    save_latent = os.path.join(result_path, args['root'] + "_latent_{}.png".format(title))
    
    args['save_best_dir'] = save_best_dir
    args['save_last_dir'] = save_last_dir
    args['save_result_dir'] = save_result_dir
    args['save_txt'] = save_txt
    args['save_conf'] = save_conf
    args['save_latent'] = save_latent
    
    return    


def process(args : Dict = DEFAULT_ARGS, dist : int = 0):
    
    # torch cuda initialize and clear cache
    torch.cuda.init()
    torch.cuda.empty_cache()

    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'

    lr = args['lr']
    seq_len = args['seq_len']
    gamma = args['gamma']
    save_best_dir = args['save_best_dir']
    save_last_dir = args['save_last_dir']
    save_conf = args["save_conf"]
    save_txt = args['save_txt']
    save_latent = args['save_latent']
    col_dim = args['col_dim']
    conv_dim = args['conv_dim']
    conv_kernel = args['conv_kernel']
    conv_stride = args['conv_stride']
    conv_padding = args['conv_padding']
    lstm_dim = args['lstm_dim']
    n_classes = args['n_classes']
    mlp_dim = args['mlp_dim']
    
    root_name = args["root"]
    
    train_data = DatasetFor0D(ts_train, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = 1.0 / 210 * 4)
    valid_data = DatasetFor0D(ts_valid, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = 1.0 / 210 * 4)
    test_data = DatasetFor0D(ts_test, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = 1.0 / 210 * 4)

    if args["use_sampler"]:
        train_sampler = ImbalancedDatasetSampler(train_data)
        valid_sampler = None
        test_sampler = None

    else:
        train_sampler = None
        valid_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_data, batch_size = args['batch_size'], sampler=train_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    valid_loader = DataLoader(valid_data, batch_size = args['batch_size'], sampler=valid_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    test_loader = DataLoader(test_data, batch_size = args['batch_size'], sampler=test_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])

    model = ConvLSTM(
        seq_len = seq_len,
        col_dim = col_dim,
        conv_dim = conv_dim,
        conv_kernel = conv_kernel,
        conv_stride = conv_stride,
        conv_padding = conv_padding,
        lstm_dim = lstm_dim,
        n_classes = n_classes,
        mlp_dim = mlp_dim
    )
    
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=gamma)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma=gamma)

    train_data.get_num_per_cls()
    cls_num_list = train_data.get_cls_num_list()

    if args['use_weight']:
        per_cls_weights = 1.0 / np.array(cls_num_list)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    else:
        per_cls_weights = np.array([1,1])
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)

    if args['use_focal_loss']:
        focal_gamma = 2.0
        loss_fn = FocalLoss(weight = per_cls_weights, gamma = focal_gamma)

    elif args['use_LDAM_loss']:
        max_m = 0.5
        s = 1.0
        loss_fn = LDAMLoss(cls_num_list, max_m = max_m, weight = per_cls_weights, s = s)
    else: 
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "sum", weight = per_cls_weights)

    if args['use_DRW']:
        betas = [0, 0.25, 0.75, 0.9]
        train_loss,  train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train_DRW(
            train_loader,
            valid_loader,
            model,
            optimizer,
            loss_fn,
            device,
            args['num_epoch'],
            args['verbose'],
            save_best_dir,
            save_last_dir,
            1.0,
            "f1_score",
            cls_num_list,
            betas
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
            max_norm_grad = 1.0,
            criteria = "f1_score",
        )

    model.load_state_dict(torch.load(save_best_dir))

    # evaluation process
    test_loss, test_acc, test_f1 = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
        save_conf = save_conf,
        save_txt = save_txt
    )
    
    # visualize latent vector
    try:
        visualize_3D_latent_space(
            model, 
            train_loader,
            device,
            save_latent
        )
    except:
        print("nan error occur")

    model.cpu()

    gc.collect()
    del train_data, valid_data, test_data
    del train_loader, valid_loader, test_loader
    del model, optimizer, scheduler

root_dir_list = ["dur21_dis0", "dur21_dis1", "dur21_dis2", "dur21_dis3", "dur21_dis4", "dur21_dis5", "dur21_dis6"]
dist_list = [0,1,2,3,4,5,6]
if __name__ == "__main__":

    kwargs = DEFAULT_ARGS

    for root_dir, dist in zip(root_dir_list, dist_list):

        kwargs['root'] = root_dir
        for idx in [0,1]:

            scheduling(kwargs, idx, args['loss_type'])
            process(kwargs, dist)
