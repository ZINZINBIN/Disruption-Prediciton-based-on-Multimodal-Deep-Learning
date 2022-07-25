import torch
import argparse
from src.dataloader import VideoDataset
from torch.utils.data import DataLoader
from src.models.resnet import Bottleneck3D
from src.models.slowfast import SlowFastClassifier, SlowFastEncoder, SlowFast
from src.train import train, train_LDAM_process
from src.evaluate import evaluate
from src.loss import FocalLossLDAM

parser = argparse.ArgumentParser(description="training SlowFast Disruption Classifier")

parser.add_argument("--batch_size", type = int, default = 16)
parser.add_argument("--lr", type = float, default = 5e-4)
parser.add_argument("--gamma", type = float, default = 0.999)
parser.add_argument("--gpu_num", type = int, default = 1)

parser.add_argument("--alpha", type = int, default = 4)
parser.add_argument("--clip_len", type = int, default = 20)

parser.add_argument("--hidden", type = int, default = 128)
parser.add_argument("--wandb_save_name", type = str, default = "slowfast-exp001")
parser.add_argument("--num_epoch", type = int, default = 16)
parser.add_argument("--verbose", type = int, default = 1)
parser.add_argument("--save_best_dir", type = str, default = "./weights/slowfast_clip_21_dist_0_best.pt")
parser.add_argument("--save_last_dir", type = str, default = "./weights/slowfast_clip_21_dist_0_last.pt")
parser.add_argument("--save_result_dir", type = str, default = "./results/train_valid_loss_acc_slowfast_clip_21_dist_0.png")
parser.add_argument("--save_test_result", type = str, default = "./results/test_slowfast_clip_21_dist_0.txt")
parser.add_argument("--use_focal_loss", type = bool, default = True)
parser.add_argument("--dataset", type = str, default = "dur0.1_dis0") # fast_model_dataset, dur0.2_dis100

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
    image_size = 224
    tau_fast = 1
    focal_gamma = 0.5
    alpha = args['alpha']
    lr = args['lr']
    mlp_hidden = args['hidden']
    num_epoch = args['num_epoch']
    verbose = args['verbose']
    gamma = args['gamma']
    save_best_dir = args['save_best_dir']
    save_last_dir = args['save_last_dir']
    save_result_dir = args["save_result_dir"]
    dataset = args["dataset"]

    train_data = VideoDataset(dataset = dataset, split = "train", clip_len = clip_len, preprocess = False, augmentation=True)
    valid_data = VideoDataset(dataset = dataset, split = "val", clip_len = clip_len, preprocess = False, augmentation=False)
    test_data = VideoDataset(dataset = dataset, split = "test", clip_len = clip_len, preprocess = False, augmentation=False)
    
    train_loader = DataLoader(train_data, batch_size = batch_size, sampler=None, num_workers = 8)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, sampler=None, num_workers = 8)
    test_loader = DataLoader(test_data, batch_size = batch_size, sampler=None, num_workers = 8)


    model = SlowFast(
        input_shape = (3, clip_len, image_size, image_size),
        block = Bottleneck3D,
        layers = [1,2,2,1],
        alpha = alpha,
        tau_fast = tau_fast,
        mlp_hidden = mlp_hidden,
        num_classes = 2,
        device = device
    )
    
    model.summary()

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=gamma)
    loss_fn = FocalLossLDAM(weight = None, gamma = focal_gamma)

    train_data.get_img_num_per_cls()
    cls_num_list = train_data.get_cls_num_list()

    train_loss,  train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train_LDAM_process(
        train_loader,
        valid_loader,
        model,
        optimizer,
        loss_fn,
        device,
        num_epoch,
        verbose,
        save_best_only = False,
        save_best_dir = save_best_dir,
        save_last_dir = save_last_dir,
        max_norm_grad = 1.0,
        criteria = "f1_score",
        cls_num_list = cls_num_list,
        gamma = focal_gamma
    )
    

    model.load_state_dict(torch.load(save_best_dir))

    # evaluation process
    test_loss, test_acc = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
        save_dir = args["save_test_result"]
    )