import torch
import argparse
import numpy as np
import gc
import os
from src.CustomDataset import CustomDataset
from src.models.ViViT import ViViT
from src.utils.sampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from typing import Dict
from src.train import train, train_DRW
from src.evaluate import evaluate
from src.loss import LDAMLoss, FocalLoss

parser = argparse.ArgumentParser(description="experiment for ViViT")
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--loss_type", type = str, default = 'LDAM')
args = vars(parser.parse_args())

DEFAULT_ARGS = {
    "batch_size":64,
    "lr" : 1e-3,
    "gamma" : 0.95,
    "gpu_num" : args['gpu_num'],
    "image_size" : 128,
    "patch_size" : 32,
    "num_workers" : 8,
    "pin_memory" : False,
    "seq_len" : 21,
    "use_sampler" : True,
    "num_epoch" : 16,
    "verbose" : 8,
    "save_best_dir" : "./weights/ViViT_clip_21_dist_5_best.pt",
    "save_last_dir" : "./weights/ViViT_clip_21_dist_5_last.pt",
    "save_result_dir" : "./results/train_valid_loss_acc_ViViT_clip_21_dist_5.png",
    "save_txt" : "./results/test_ViViT_clip_21_dist_5.txt",
    "save_conf" : "./results/test_ViViT_clip_21_dist_5_confusion_matrix.png",
    "use_focal_loss" : True if args['loss_type'] == 'FOCAL' else False,
    "use_LDAM_loss" : True if args['loss_type'] == 'LDAM' else False,
    "use_weight" : False,
    "root_dir" : "./dataset/dur21_dis0",
    "use_DRW" : False,
}

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

def scheduling(args : Dict, idx : int, loss_type : str):

    weight_path = os.path.join("./weights", "experiment_{}".format(loss_type)) 
    result_path = os.path.join("./results", "experiment_{}".format(loss_type))  

    if idx == 0:
        args['use_sampler'] = False
        args['use_weight'] = False
        args['use_DRW'] = False
        title = "BS"

    elif idx == 1:
        args['use_sampler'] = True
        args['use_weight'] = False
        args['use_DRW'] = False
        title = "RS"

    elif idx == 2:
        args['use_sampler'] = False
        args['use_weight'] = True
        args['use_DRW'] = False
        title = "RW"

    elif idx == 3:
        args['use_sampler'] = True
        args['use_weight'] = True
        args['use_DRW'] = False
        title = "RS_RW"

    elif idx == 4:
        args['use_sampler'] = False
        args['use_weight'] = False
        args['use_DRW'] = True
        title = "DRW"
    
    else:
        args['use_sampler'] = True
        args['use_weight'] = False
        args['use_DRW'] = True
        title = "DRW_RS"

    save_best_dir = os.path.join(weight_path, args['root_dir'].split("/")[-1] + "_{}.pt".format(title))
    save_last_dir = os.path.join(weight_path, args['root_dir'].split("/")[-1] + "_{}.pt".format(title))
    save_result_dir = os.path.join(result_path, args['root_dir'].split("/")[-1] + "_loss_curve_{}.png".format(title))
    save_txt = os.path.join(result_path, args['root_dir'].split("/")[-1] + "_eval_{}.txt".format(title))
    save_conf = os.path.join(result_path, args['root_dir'].split("/")[-1] + "_confusion_{}.png".format(title))

    args['save_best_dir'] = save_best_dir
    args['save_last_dir'] = save_last_dir
    args['save_result_dir'] = save_result_dir
    args['save_txt'] = save_txt
    args['save_conf'] = save_conf

    return    


def process(args : Dict = DEFAULT_ARGS):
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

    model.cpu()

    gc.collect()
    del train_data, valid_data, test_data
    del train_loader, valid_loader, test_loader
    del model, optimizer, scheduler

root_dir_list = ["./dataset/dur21_dis0", "./dataset/dur21_dis3", "./dataset/dur21_dis5"]

if __name__ == "__main__":

    kwargs = DEFAULT_ARGS

    for root_dir in root_dir_list:

        kwargs['root_dir'] = root_dir

        if args['loss_type'] not in ['LDAM', 'FOCAL']:
            n = 4
        else:
            n = 6

        # for idx in range(n):
        for idx in [6]:

            scheduling(kwargs, idx, args['loss_type'])
            process(kwargs)
