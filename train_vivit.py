import torch
import argparse
import matplotlib.pyplot as plt
from src.dataloader import VideoDataset
from src.models.ViViT import ViViT
from src.utils.sampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from src.train import train
from src.evaluate import evaluate
from src.loss import FocalLoss, LDAMLoss, FocalLossLDAM

parser = argparse.ArgumentParser(description="training ViViT for disruption classifier")
parser.add_argument("--batch_size", type = int, default = 64)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--gamma", type = float, default = 0.995)
parser.add_argument("--gpu_num", type = int, default = 0)

parser.add_argument("--num_workers", type = int, default = 8)
parser.add_argument("--pin_memory", type = bool, default = True)

parser.add_argument("--use_sampler", type = bool, default = False)
parser.add_argument("--wandb_save_name", type = str, default = "ViViT-exp001")
parser.add_argument("--num_epoch", type = int, default = 8)
parser.add_argument("--verbose", type = int, default = 1)
parser.add_argument("--save_best_dir", type = str, default = "./weights/ViViT_clip_42_dist_21_best.pt")
parser.add_argument("--save_last_dir", type = str, default = "./weights/ViViT_clip_42_dist_21_last.pt")
parser.add_argument("--save_result_dir", type = str, default = "./results/train_valid_loss_acc_ViViT_clip_42_dist_21.png")
parser.add_argument("--save_test_result", type = str, default = "./results/test_ViViT_clip_42_dist_21.txt")
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
    lr = args['lr']
    clip_len = 42
    num_epoch = args['num_epoch']
    verbose = args['verbose']
    gamma = args['gamma']
    save_best_dir = args['save_best_dir']
    save_last_dir = args['save_last_dir']
    save_result_dir = args["save_result_dir"]
    dataset = args["dataset"]

    train_data = VideoDataset(dataset = args["dataset"], split = "train", clip_len = clip_len, preprocess = False, augmentation = True)
    valid_data = VideoDataset(dataset = args["dataset"], split = "val", clip_len = clip_len, preprocess = False, augmentation=False)
    test_data = VideoDataset(dataset = args["dataset"], split = "test", clip_len = clip_len, preprocess = False, augmentation=False)

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

    sample_x, sample_y = next(iter(train_loader))
    print("sample_x : ", sample_x.size())
    print("sample_y : ", sample_y.size())

    model = ViViT(
        image_size = 112,
        patch_size = 16,
        n_classes = 2,
        n_frames = clip_len,
        dim = 128,
        depth = 4,
        n_heads = 4,
        pool = "cls",
        in_channels = 3,
        d_head = 64,
        dropout = 0.25,
        embedd_dropout=0.25,
        scale_dim = 4
    )

    model.to(device)

    model.summary(device, show_input = True, show_hierarchical=False, print_summary=True, show_parent_layers=False)

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
        save_best_only = False,
        save_best_dir = save_best_dir,
        save_last_dir = save_last_dir,
        use_video_mixup_algorithm = False,
        max_norm_grad = 1.0,
        criteria = "f1_score"
        )

    model.load_state_dict(torch.load(save_best_dir))

    test_loss, test_acc = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
        save_dir = args["save_test_result"]
    )