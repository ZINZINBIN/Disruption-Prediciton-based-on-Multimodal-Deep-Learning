import torch
import argparse
import matplotlib.pyplot as plt
from src.dataloader import VideoDataset
from torch.utils.data import DataLoader
from src.models.R2Plus1DwithSTN import R2P1DwithSTN, R2P1DwithSTNClassifier
from src.train import train
from src.evaluate import evaluate
from src.loss import FocalLossLDAM

parser = argparse.ArgumentParser(description="training R2Plus1D with STN model")
parser.add_argument("--batch_size", type = int, default = 16)
parser.add_argument("--lr", type = float, default = 5e-4)
parser.add_argument("--gamma", type = float, default = 0.999)
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--alpha", type = float, default = 0.01)
parser.add_argument("--clip_len", type = int, default = 11)
parser.add_argument("--wandb_save_name", type = str, default = "R2Plus1D_STN-exp001")
parser.add_argument("--num_epoch", type = int, default = 256)
parser.add_argument("--verbose", type = int, default = 2)
parser.add_argument("--save_best_dir", type = str, default = "./weights/R2P1D_STN_best.pt")
parser.add_argument("--save_last_dir", type = str, default = "./weights/R2P1D_STN_last.pt")
parser.add_argument("--save_result_dir", type = str, default = "./results/train_valid_loss_acc_R2P1D_STN.png")
parser.add_argument("--use_focal_loss", type = bool, default = False)
parser.add_argument("--dataset", type = str, default = "dur0.1_dis10") # fast_model_dataset, dur0.2_dis100

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
    save_last_dir = args['save_last_dir']
    save_result_dir = args["save_result_dir"]
    dataset = args['dataset']

    train_data= VideoDataset(dataset = dataset, split = "train", clip_len = clip_len, preprocess = False, augmentation=True)
    valid_data = VideoDataset(dataset = dataset, split = "val", clip_len = clip_len, preprocess = False, augmentation=False)
    test_data = VideoDataset(dataset = dataset, split = "test", clip_len = clip_len, preprocess = False, augmentation=False)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 4)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True, num_workers = 4)

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
        loss_fn = FocalLossLDAM(weight = None, gamma = 0.5)
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
        num_epoch,
        verbose,
        save_best_only = False,
        save_best_dir = save_best_dir,
        save_last_dir = save_last_dir,
        use_video_mixup_algorithm = False,
        max_norm_grad = 1.0,
        criteria = "f1_score"
    )


    test_loss, test_acc = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
        save_dir = "./results/test_R2P1D_STN.txt"
    )