from sched import scheduler
from typing import Optional
import torch
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from src.dataloader import VideoDataset
from torch.utils.data import DataLoader
from src.R2Plus1DwithSTN import *
from src.train import train
from src.evaluate import evaluate
from src.loss import FocalLoss
from tqdm import tqdm

parser = argparse.ArgumentParser(description="training R2Plus1D with STN model")
parser.add_argument("--batch_size", type = int, default = 12)
parser.add_argument("--lr", type = float, default = 5e-4)
parser.add_argument("--gamma", type = float, default = 0.999)
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--alpha", type = float, default = 0.01)
parser.add_argument("--clip_len", type = int, default = 10)
parser.add_argument("--wandb_save_name", type = str, default = "R2Plus1D_STN-exp001")
parser.add_argument("--num_epoch", type = int, default = 256)
parser.add_argument("--verbose", type = int, default = 4)
parser.add_argument("--save_best_dir", type = str, default = "./weights/R2P1D_STN_best.pt")
parser.add_argument("--save_result_dir", type = str, default = "./results/train_valid_loss_acc_R2P1D_STN.png")
parser.add_argument("--use_focal_loss", type = bool, default = False)

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
    device = "cuda:0"
else:
    device = 'cpu'

if __name__ == "__main__":

    batch_size = args['batch_size']
    clip_len = args['clip_len']
    alpha = args['alpha']
    lr = args['lr']
    num_epoch = args['num_epoch']
    verbose = args['verbose']
    save_best_dir = args['save_best_dir']
    save_result_dir = args["save_result_dir"]

    train_data_dist10 = VideoDataset(dataset = "fast_model_dataset", split = "train", clip_len = clip_len, preprocess = False)
    valid_data_dist10 = VideoDataset(dataset = "fast_model_dataset", split = "val", clip_len = clip_len, preprocess = False)
    test_data_dist10 = VideoDataset(dataset = "fast_model_dataset", split = "test", clip_len = clip_len, preprocess = False)

    train_loader_dist10 = DataLoader(train_data_dist10, batch_size = batch_size, shuffle = True, num_workers = 4)
    valid_loader_dist10 = DataLoader(valid_data_dist10, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader_dist10 = DataLoader(test_data_dist10, batch_size = batch_size, shuffle = True, num_workers = 4)

    model = R2P1DwithSTNClassifier(
        input_size  = (3, clip_len, 112, 112),
        num_classes = 2, 
        layer_sizes = [2,2,2,2], 
        pretrained = False, 
        alpha = alpha
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0 = 8,
        T_mult = 2
    )

    if args['use_focal_loss']:
        loss_fn = FocalLoss(alpha = 0.25, gamma=2, size_average=True)
    else: 
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")

    train_loss,  train_acc, valid_loss, valid_acc = train(
        train_loader_dist10,
        valid_loader_dist10,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        num_epoch,
        verbose,
        save_best_only=True,
        save_best_dir = save_best_dir
    )

    test_loss, test_acc = evaluate(
        test_loader_dist10,
        model,
        optimizer,
        loss_fn,
        device
    )
    
    x_axis = range(1, num_epoch + 1)
    plt.figure(figsize = (16,10))
    plt.subplot(1,2,1)
    plt.plot(x_axis, train_loss, 'ro-', label  = "train loss")
    plt.plot(x_axis, valid_loss, 'b^-', label = "valid loss")
    plt.xlabel("epoch")
    plt.ylabel("loss(CrossEntropy)")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(x_axis, train_acc, 'ro--', label = "train acc")
    plt.plot(x_axis, valid_acc, 'bo--', label = 'valid acc')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(save_result_dir)