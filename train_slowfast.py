from sched import scheduler
from typing import Optional
import torch
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from src.dataloader import VideoDataset
from torch.utils.data import DataLoader
from src.models.model import SlowFastDisruptionClassifier
from src.models.resnet import Bottleneck3D
from src.train import train
from src.evaluate import evaluate
from src.loss import FocalLoss
from src.utils.sampler import ImbalancedDatasetSampler
from tqdm import tqdm

parser = argparse.ArgumentParser(description="training SlowFast Disruption Classifier")
parser.add_argument("--batch_size", type = int, default = 24)
parser.add_argument("--lr", type = float, default = 5e-4)
parser.add_argument("--gamma", type = float, default = 0.999)
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--alpha", type = int, default = 4)
parser.add_argument("--p", type = float, default = 0.5)
parser.add_argument("--clip_len", type = int, default = 42)
parser.add_argument("--hidden", type = int, default = 128)
parser.add_argument("--wandb_save_name", type = str, default = "slowfast-exp001")
parser.add_argument("--num_epoch", type = int, default = 256)
parser.add_argument("--verbose", type = int, default = 1)
parser.add_argument("--save_best_dir", type = str, default = "./weights/slowfast_clip_42_dist_21_best.pt")
parser.add_argument("--save_result_dir", type = str, default = "./results/train_valid_loss_acc_slowfast_clip_42_dist_21.png")
parser.add_argument("--save_test_result", type = str, default = "./results/test_slowfast_clip_42_dist_21.txt")
parser.add_argument("--use_focal_loss", type = bool, default = False)
parser.add_argument("--dataset", type = str, default = "dur0.2_dis21") # fast_model_dataset, dur0.2_dis100

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
    
# dataset composition
import os
try:
    path = "./dataset/" + args["dataset"] + "/"
    path_disruption = path + "disruption/"
    path_borderline = path + "borderline/"
    path_normal = path + "normal/"

    dir_disruption_list = os.listdir(path_disruption)
    dir_borderline_list = os.listdir(path_borderline)
    dir_normal_list = os.listdir(path_normal)

    print("disruption : ", len(dir_disruption_list))
    print("normal : ", len(dir_normal_list))
    print("borderline : ", len(dir_borderline_list))
except:
    print("video dataset directory is not valid")

if __name__ == "__main__":

    batch_size = args['batch_size']
    clip_len = args['clip_len']
    alpha = args['alpha']
    lr = args['lr']
    p = args['p']
    hidden = args['hidden']
    num_epoch = args['num_epoch']
    verbose = args['verbose']
    gamma = args['gamma']
    save_best_dir = args['save_best_dir']
    save_result_dir = args["save_result_dir"]
    dataset = args["dataset"]

    train_data_dist = VideoDataset(dataset = dataset, split = "train", clip_len = clip_len, preprocess = False)
    valid_data_dist = VideoDataset(dataset = dataset, split = "val", clip_len = clip_len, preprocess = False)
    test_data_dist = VideoDataset(dataset = dataset, split = "test", clip_len = clip_len, preprocess = False)
    
    train_sampler = ImbalancedDatasetSampler(train_data_dist)
    valid_sampler = ImbalancedDatasetSampler(valid_data_dist)
    test_sampler = ImbalancedDatasetSampler(test_data_dist)
    
    train_loader_dist = DataLoader(train_data_dist, batch_size = batch_size, sampler=train_sampler, num_workers = 8)
    valid_loader_dist = DataLoader(valid_data_dist, batch_size = batch_size, sampler=valid_sampler, num_workers = 8)
    test_loader_dist = DataLoader(test_data_dist, batch_size = batch_size, sampler=test_sampler, num_workers = 8)

    model = SlowFastDisruptionClassifier(
        input_shape = (3,clip_len,112,112),
        block = Bottleneck3D,
        layers = [1,2,2,1], #[1,1,1,1], #[3,4,6,3],
        alpha = alpha,
        p = p,
        mlp_hidden = hidden,
        num_classes  = 2
    )
    
    model.summary()

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0 = 8,
        T_mult = 2
    )

    if args['use_focal_loss']:
        loss_fn = FocalLoss(alpha = 1.0, gamma=2, size_average=True)
    else: 
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")

    train_loss,  train_acc, valid_loss, valid_acc = train(
        train_loader_dist,
        valid_loader_dist,
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

    model.load_state_dict(torch.load(save_best_dir))

    test_loss, test_acc = evaluate(
        test_loader_dist,
        model,
        optimizer,
        loss_fn,
        device,
        save_dir = args["save_test_result"]
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