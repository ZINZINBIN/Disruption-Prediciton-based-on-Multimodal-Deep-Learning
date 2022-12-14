import torch
import argparse
import numpy as np
import pandas as pd
from src.CustomDataset import DatasetFor0D
from src.models.ConvLSTM import ConvLSTM
from src.utils.sampler import ImbalancedDatasetSampler
from src.utils.utility import plot_learning_curve, generate_prob_curve_from_0D, preparing_0D_dataset
from src.visualization.visualize_latent_space import visualize_2D_latent_space, visualize_3D_latent_space
from torch.utils.data import DataLoader
from src.train import train
from src.evaluate import evaluate
from src.loss import LDAMLoss, FocalLoss
from torch.utils.data import DataLoader
from src.utils.sampler import ImbalancedDatasetSampler

# columns for use
ts_cols = ['\\q95', '\\ipmhd', '\\kappa', '\\tritop', '\\tribot','\\betap','\\betan','\\li', '\\WTOT_DLM03']

# parsing
parser = argparse.ArgumentParser(description="training 1D CNN - LSTM for disruption classifier")
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--gpu_num", type = int, default = 2)

parser.add_argument("--num_workers", type = int, default = 8)
parser.add_argument("--pin_memory", type = bool, default = False)

parser.add_argument("--dist", type = int, default = 3)
parser.add_argument("--dt", type = float, default = 4 * 1 / 210)
parser.add_argument("--seq_len", type = int, default = 21)
parser.add_argument("--use_sampler", type = bool, default = True)
parser.add_argument("--num_epoch", type = int, default = 64)
parser.add_argument("--verbose", type = int, default = 4)

parser.add_argument("--save_best_dir", type = str, default = "./weights/ts_conv_lstm_clip_21_dist_3_best.pt")
parser.add_argument("--save_last_dir", type = str, default = "./weights/ts_conv_lstm_clip_21_dist_3_last.pt")
parser.add_argument("--save_txt", type = str, default = "./results/test_ts_conv_lstm_clip_21_dist_3.txt")
parser.add_argument("--save_conf", type = str, default = "./results/test_ts_conv_lstm_clip_21_dist_3_confusion_matrix.png")
parser.add_argument("--save_latent_2d", type = str, default = "./results/ts_conv_lstm_clip_21_dist_3_latent_2d.png")
parser.add_argument("--save_latent_3d", type = str, default = "./results/ts_conv_lstm_clip_21_dist_3_latent_3d.png")

parser.add_argument("--use_focal_loss", type = bool, default = True)
parser.add_argument("--use_LDAM_loss", type = bool, default = False)
parser.add_argument("--use_weight", type = bool, default = True)

args = vars(parser.parse_args())

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:" + str(args["gpu_num"])
else:
    device = 'cpu'
    
if __name__ == "__main__":
    
    # parsing
    batch_size = args['batch_size']
    lr = args['lr']
    num_workers = args['num_workers']
    gamma = args['gamma']
    num_epoch = args['num_epoch']
    verbose = args['verbose']
    seq_len = args['seq_len']
    dist = args['dist']
    dt = args['dt']
    col_len = len(ts_cols)
    save_best_dir = args['save_best_dir']
    save_last_dir = args['save_last_dir']
    save_txt = args['save_txt']
    save_conf = args['save_conf']
    save_latent_2d = args['save_latent_2d']
    save_latent_3d = args['save_latent_3d']
    
    # dataset
    ts_train, ts_valid, ts_test, ts_scaler = preparing_0D_dataset("./dataset/KSTAR_Disruption_ts_data_extend.csv", ts_cols = ts_cols, scaler = 'Robust')
    
    # disruption info
    kstar_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List.csv', encoding = "euc-kr")

    train_data = DatasetFor0D(ts_train, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = dt)
    valid_data = DatasetFor0D(ts_valid, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = dt)
    test_data = DatasetFor0D(ts_test, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = dt)
    
    if args["use_sampler"]:
        train_sampler = ImbalancedDatasetSampler(train_data)
        valid_sampler = None
        test_sampler = None

    else:
        train_sampler = None
        valid_sampler = None
        test_sampler = None
        
    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers =num_workers, sampler = train_sampler)
    
    if valid_sampler:
        valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = num_workers, sampler = valid_sampler)
    else:
        valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = num_workers, shuffle = True)
    
    if test_sampler:
        test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = num_workers, sampler = test_sampler)
    else:
        test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = num_workers, shuffle = True)    
    
    # model
    model = ConvLSTM(
        seq_len = seq_len,
        col_dim = col_len,
    )
    
    model.summary()

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
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean", weight = per_cls_weights)
        
    train_loss,  train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        num_epoch,
        verbose,
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

    # plot probability curve
    generate_prob_curve_from_0D(
        model, 
        batch_size = 1, 
        device = device, 
        save_dir = "./results/disruption_probs_curve.png",
        ts_data = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
        ts_cols = ts_cols,
        shot_list_dir = './dataset/KSTAR_Disruption_Shot_List_extend.csv',
        shot = 21310,
        seq_len = seq_len,
        dist = dist,
        dt = dt
    )
    
    # plot 2d latent space
    visualize_2D_latent_space(
        model,
        train_loader,
        device,
        save_latent_2d
    )
    
    # plot 3d latent space
    visualize_3D_latent_space(
        model,
        train_loader,
        device,
        save_latent_3d
    )