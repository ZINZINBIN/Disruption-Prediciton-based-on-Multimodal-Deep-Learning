from sched import scheduler
from typing import Optional
import torch
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from src.dataloader import VideoDataset
from torch.utils.data import DataLoader
from src.model import SlowFastDisruptionClassifier
from src.resnet import Bottleneck3D
from src.train import train
from src.evaluate import evaluate
from src.loss import FocalLoss
from src.distributed import train_distributed
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Distributed training code")
parser.add_argument("--batch_size", type = int, default = 12)
parser.add_argument("--lr", type = float, default = 2e-4)
parser.add_argument("--gamma", type = float, default = 0.999)
parser.add_argument("--alpha", type = int, default = 2)
parser.add_argument("--p", type = float, default = 0.5)
parser.add_argument("--clip_len", type = int, default = 8)
parser.add_argument("--hidden", type = int, default = 128)
parser.add_argument("--num_epoch", type = int, default = 248)
parser.add_argument("--verbose", type = int, default = 4)
parser.add_argument("--save_best_dir", type = str, default = "./weights/slowfast_clip_10_best.pt")
parser.add_argument("--save_dir", type = str, default = "./weights/slowfast_clip_10.pt")
parser.add_argument("--save_result_dir", type = str, default = "./results/train_valid_loss_acc_slowfast_clip_10.png")
parser.add_argument("--save_test_result", type = str, default = "./results/test_slowfast_clip_10.txt")
parser.add_argument("--use_focal_loss", type = bool, default = False)

args = vars(parser.parse_args())

# torch device state
# print("torch device avaliable : ", torch.cuda.is_available())
# print("torch current device : ", torch.cuda.current_device())
# print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":

    import os

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    batch_size = args['batch_size']
    clip_len = args['clip_len']
    alpha = args['alpha']
    lr = args['lr']
    p = args['p']
    hidden = args['hidden']
    num_epoch = args['num_epoch']
    verbose = args['verbose']
    gamma = args['gamma']
    save_dir = args["save_dir"]
    save_best_dir = args['save_best_dir']
    save_result_dir = args["save_result_dir"]

    train_data_dist10 = VideoDataset(dataset = "fast_model_dataset", split = "train", clip_len = clip_len, preprocess = False)
    valid_data_dist10 = VideoDataset(dataset = "fast_model_dataset", split = "val", clip_len = clip_len, preprocess = False)
    test_data_dist10 = VideoDataset(dataset = "fast_model_dataset", split = "test", clip_len = clip_len, preprocess = False)

    train_loader_dist10 = DataLoader(train_data_dist10, batch_size = batch_size, shuffle = True, num_workers = 4)
    valid_loader_dist10 = DataLoader(valid_data_dist10, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader_dist10 = DataLoader(test_data_dist10, batch_size = batch_size, shuffle = True, num_workers = 4)

    model = SlowFastDisruptionClassifier(
        input_shape = (3,clip_len,112,112),
        block = Bottleneck3D,
        layers = [1,2,2,1], #[3,4,6,3],
        alpha = alpha,
        p = p,
        mlp_hidden = hidden,
        num_classes  = 2
    )

    if args['use_focal_loss']:
        loss_fn = FocalLoss(alpha = 0.25, gamma=2, size_average=True)
    else: 
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")

    # 분산 학습 진행
    train_distributed(
        world_size = 1, #torch.cuda.device_count(),
        batch_size = batch_size,
        model = model,
        train_dataset=train_data_dist10,
        valid_dataset=valid_data_dist10,
        random_seed = 42,
        resume = True,
        loss_fn = loss_fn,
        model_filepath = save_dir,
        num_epoch=num_epoch,
        verbose = verbose,
        save_best_only=True,
        save_best_dir=save_best_dir
    )
    
    test_loss, test_acc = evaluate(
        test_loader_dist10,
        model,
        None,
        loss_fn,
        None,
        save_dir = args["save_test_result"]
    )