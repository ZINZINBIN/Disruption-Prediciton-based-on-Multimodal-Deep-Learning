import torch
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import DataLoader
from src.CustomDataset import DEFAULT_TS_COLS, MultiModalDataset
from src.utils.sampler import ImbalancedDatasetSampler
from src.train import train
from src.evaluate import evaluate
from src.loss import LDAMLoss, FocalLoss
from src.GradientBlending import GradientBlending, train_GB_dynamic
from src.models.MultiModal import MultiModalModel, FusionNetwork, MultiModalNetwork

parser = argparse.ArgumentParser(description="training multimodal network for disruption classifier")
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--gpu_num", type = int, default = 0)

parser.add_argument("--num_workers", type = int, default = 8)
parser.add_argument("--pin_memory", type = bool, default = False)

parser.add_argument("--use_sampler", type = bool, default = True)
parser.add_argument("--num_epoch", type = int, default = 64)
parser.add_argument("--verbose", type = int, default = 1)
parser.add_argument("--save_best_dir", type = str, default = "./weights/multi_modal_clip_21_dist_8_best.pt")
parser.add_argument("--save_last_dir", type = str, default = "./weights/multi_modal_clip_21_dist_8_last.pt")
parser.add_argument("--save_result_dir", type = str, default = "./results/train_valid_loss_acc_multi_modal_clip_21_dist_8.png")
parser.add_argument("--save_txt", type = str, default = "./results/test_multi_modal_clip_21_dist_8.txt")
parser.add_argument("--save_conf", type = str, default = "./results/test_multi_modal_clip_21_dist_8_confusion_matrix.png")
parser.add_argument("--save_latent_dir", type = str, default = "./results/multi_modal_clip_21_dist_8_2d_latent.png")
parser.add_argument("--use_focal_loss", type = bool, default = True)
parser.add_argument("--use_LDAM_loss", type = bool, default = False)
parser.add_argument("--use_weight", type = bool, default = True)
parser.add_argument("--root_dir", type = str, default = "./dataset/dur84_dis8")

args = vars(parser.parse_args())

# default argument
args_video = {
    "image_size" : 128, 
    "patch_size" : 32, 
    "n_frames" : 21, 
    "dim": 64 * 2, 
    "depth" : 4, 
    "n_heads" : 8, 
    "pool" : 'cls', 
    "in_channels" : 3, 
    "d_head" : 64, 
    "dropout" : 0.25,
    "embedd_dropout":  0.25, 
    "scale_dim" : 4
}

args_0D = {
    "seq_len" : 21, 
    "col_dim" : 9, 
    "conv_dim" : 32, 
    "conv_kernel" : 3,
    "conv_stride" : 1, 
    "conv_padding" : 1,
    "lstm_dim" : 64, 
}

args_fusion = {
    "kernel_size" : 3,
    "stride" : 2,
    "maxpool_kernel" : 3,
    "maxpool_stride" : 2,
}

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
    
    kstar_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_extend.csv', encoding = "euc-kr")
    ts_data = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_for_multi.csv")
    mult_info = pd.read_csv("./dataset/KSTAR_Disruption_multi_data.csv")

    train_data = MultiModalDataset('train', ts_data, DEFAULT_TS_COLS, mult_info, dt = 1 / 210 * 4, distance = 8, seq_len = 21)
    valid_data = MultiModalDataset('valid', ts_data, DEFAULT_TS_COLS, mult_info, dt = 1 / 210 * 4, distance = 8, seq_len = 21)
    test_data = MultiModalDataset('test', ts_data, DEFAULT_TS_COLS, mult_info, dt = 1 / 210 * 4, distance = 8, seq_len = 21)

    batch_size = args["batch_size"]
    lr = args['lr']

    if args["use_sampler"]:
        train_sampler = ImbalancedDatasetSampler(train_data)
        valid_sampler = None
        test_sampler = None

    else:
        train_sampler = None
        valid_sampler = None
        test_sampler = None
        
    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers = args['num_workers'], sampler = train_sampler)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = args['num_workers'], sampler = valid_sampler)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = args['num_workers'], sampler = test_sampler)

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

    model = FusionNetwork(
        2, args_video, args_0D, args_fusion
    )

    model.summary('cpu')
    model.to(device)

    num_epoch = args['num_epoch']
    verbose = args['verbose']
    max_norm_grad = 1.0
    criteria = "f1_score"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma=args['gamma'])

    train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        num_epoch,
        verbose,
        args['save_best_dir'],
        args['save_last_dir'],
        max_norm_grad,
        criteria,
        model_type = 'multi'
    )

    model.load_state_dict(torch.load(args['save_best_dir']))

    # evaluation process
    test_loss, test_acc, test_f1 = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
        save_conf = args['save_conf'],
        save_txt = args['save_txt'],
        model_type = 'multi'
    )