import torch
import argparse
import numpy as np
from src.CustomDataset import CustomDataset
from src.models.ViViT import ViViT
from src.utils.sampler import ImbalancedDatasetSampler
from src.utils.utility import show_data_composition, plot_learning_curve
from torch.utils.data import DataLoader
from src.train import train, train_LDAM_process
from src.evaluate import evaluate
from src.loss import LDAMLoss, FocalLoss

parser = argparse.ArgumentParser(description="training ViViT for disruption classifier")
parser.add_argument("--batch_size", type = int, default = 32)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--gpu_num", type = int, default = 2)

parser.add_argument("--image_size", type = int, default = 128)
parser.add_argument("--patch_size", type = int, default = 32)

parser.add_argument("--num_workers", type = int, default = 8)
parser.add_argument("--pin_memory", type = bool, default = False)

parser.add_argument("--seq_len", type = int, default = 21)
parser.add_argument("--use_sampler", type = bool, default = True)
parser.add_argument("--wandb_save_name", type = str, default = "ViViT-exp001")
parser.add_argument("--num_epoch", type = int, default = 128)
parser.add_argument("--verbose", type = int, default = 1)
parser.add_argument("--save_best_dir", type = str, default = "./weights/ViViT_clip_21_dist_0_best.pt")
parser.add_argument("--save_last_dir", type = str, default = "./weights/ViViT_clip_21_dist_0_last.pt")
parser.add_argument("--save_result_dir", type = str, default = "./results/train_valid_loss_acc_ViViT_clip_21_dist_0.png")
parser.add_argument("--save_txt", type = str, default = "./results/test_ViViT_clip_21_dist_0.txt")
parser.add_argument("--save_conf", type = str, default = "./results/test_ViViT_clip_21_dist_0_confusion_matrix.png")
parser.add_argument("--use_focal_loss", type = bool, default = True)
parser.add_argument("--root_dir", type = str, default = "./dataset/dur21_dis0")

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
try:
    show_data_composition(args['root_dir'])
except:
    print("Directory is invalid")

if __name__ == "__main__":

    batch_size = args['batch_size']
    lr = args['lr']
    seq_len = args['seq_len']
    num_epoch = args['num_epoch']
    verbose = args['verbose']
    gamma = args['gamma']
    save_best_dir = args['save_best_dir']
    save_last_dir = args['save_last_dir']
    save_conf = args["save_conf"]
    save_txt = args['save_txt']
    root_dir = args["root_dir"]
    image_size = args['image_size']

    train_data = CustomDataset(root_dir = root_dir, task = 'train', ts_data = None, augmentation = False, crop_size = image_size, seq_len = seq_len, mode = 'video')
    valid_data = CustomDataset(root_dir = root_dir, task = 'valid', ts_data = None, augmentation = False, crop_size = image_size, seq_len = seq_len, mode = 'video')
    test_data = CustomDataset(root_dir = root_dir, task = 'test', ts_data = None, augmentation = False, crop_size = image_size, seq_len = seq_len, mode = 'video')

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

    model = ViViT(
        image_size = args['image_size'],
        patch_size = args['patch_size'],
        n_classes = 2,
        n_frames = seq_len,
        dim = 64,
        depth = 4,
        n_heads = 8,
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
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 8,T_mult = 2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma=gamma)

    if args['use_focal_loss']:
        train_data.get_num_per_cls()
        cls_num_list = train_data.get_cls_num_list()
        per_cls_weights = 1.0 / np.array(cls_num_list)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
        per_cls_weights = torch.FloatTensor(per_cls_weights)

        focal_gamma = 2.0
        loss_fn = FocalLoss(weight = per_cls_weights, gamma = focal_gamma)

    else: 
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "sum")

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

    plot_learning_curve(train_loss, valid_loss, train_f1, valid_f1, figsize = (12,6), save_dir = "./results/train_valid_loss_f1_curve_ViViT_clip_21_dist_0.png")

    '''
    # training process
    if args['use_focal_loss']:
        train_loss,  train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train_LDAM_process(
            train_loader,
            valid_loader,
            model,
            optimizer,
            loss_fn,
            device,
            args['num_epoch'],
            args['verbose'],
            save_best_dir = save_best_dir,
            save_last_dir = save_last_dir,
            max_norm_grad = 1.0,
            criteria = "f1_score",
            cls_num_list = cls_num_list,
            gamma = focal_gamma
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
    '''
    
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