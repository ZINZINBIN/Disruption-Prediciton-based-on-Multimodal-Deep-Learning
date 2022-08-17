import torch
import numpy as np
import seaborn as sns
import argparse
import os
import matplotlib.pyplot as plt
from src.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from src.utils.sampler import ImbalancedDatasetSampler
from src.train import train
from src.evaluate import evaluate
from src.loss import FocalLoss, LDAMLoss

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="training disruption prediction model")

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)

    # data input shape
    parser.add_argument("--channel", type = int, default = 3)
    parser.add_argument("--height", type = int, default = 112)
    parser.add_argument("--width", type = int, default = 112)

    # common argument
    # batch size / learning rate / clip length / epochs / dataset / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 24)
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--gamma", type = float, default = 0.999)
    parser.add_argument("--clip_len", type = int, default = 42)
    parser.add_argument("--num_epoch", type = int, default = 256)
    parser.add_argument("--dataset", type = str, default = "dur0.2_dis21")
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = True)

    # model weight / save process
    # wandb setting
    parser.add_argument("--use_tensorboard", type = bool, default = False)
    parser.add_argument("--use_wandb", type = bool, default = False)
    parser.add_argument("--wandb_save_name", type = str, default = "SBERT-exp001")

    # local save
    parser.add_argument("--save_best", type = str, default = "./weights/sbert_clip_42_dis_21_no_sampler_best.pt")
    parser.add_argument("--save_training_curve", type = str, default = "./results/train_valid_loss_acc_sbert_clip_42_dis_21_no_sampler.png")
    parser.add_argument("--save_evaluation", type = str, default = "./results/test_SBERT_clip_42_dis_21_no_sampler.txt")
    
    # detail setting for training process
    # data augmentation
    # video_mixup
    parser.add_argument("--use_video_mixup", type = bool, default = False)

    # conventional augmentation
    parser.add_argument("--bright_val", type = int, default = 30)
    parser.add_argument("--bright_p", type = float, default = 0.25)
    parser.add_argument("--contrast_min", type = float, default = 1)
    parser.add_argument("--contrast_max", type = float, default = 1.5)
    parser.add_argument("--contrast_p", type = float, default = 0.25)
    parser.add_argument("--blur_k", type = int, default = 5)
    parser.add_argument("--blur_p", type = float, default = 0.25)
    parser.add_argument("--flip_p", type = float, default = 0.25)
    parser.add_argument("--vertical_ratio", type = float, default = 0.2)
    parser.add_argument("--vertical_p", type = float, default = 0.25)
    parser.add_argument("--horizontal_ratio", type = float, default = 0.2)
    parser.add_argument("--horizontal_p", type = float, default = 0.25)

    # optimizer : SGD, RMSProps, Adam, AdamW
    parser.add_argument("--optimizer", type = str, default = "AdamW")

    # imbalanced dataset processing
    # imbalanced dataset sampler : conventional
    parser.add_argument("--use_sampler", type = bool, default = False)

    # training schedule : None, Resample, Reweight, DRW
    parser.add_argument("--train_rule", type = str, default = 'None')

    # use conventional scheduler
    parser.add_argument("--use_scheduler", type = bool, default = True)
    parser.add_argument("--T_0", type = int, default = 8)
    parser.add_argument("--T_mult", type = int, default = 2)

    # loss type : CE, Focal, LDAM
    parser.add_argument("--loss_type", type = str, default = "CE")
    
    # monitoring the training process
    parser.add_argument("--verbose", type = int, default = -1)
    
    # model setup
    parser.add_argument("--alpha", type = float, default = 0.01)
    parser.add_argument("--dropout", type = float, default = 0.5)

    args = vars(parser.parse_args())

    return args

# torch device state
print("############### device setup ###################")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":

    args = parsing()

    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'

    # dataset composition
    try:
        path = "./dataset/" + args["dataset"] + "/"
        path_disruption = path + "disruption/"
        path_borderline = path + "borderline/"
        path_normal = path + "normal/"

        dir_disruption_list = os.listdir(path_disruption)
        dir_borderline_list = os.listdir(path_borderline)
        dir_normal_list = os.listdir(path_normal)
        
        print("\n############## dataset composition ##############")
        print("disruption : ", len(dir_disruption_list))
        print("normal : ", len(dir_normal_list))
        print("borderline : ", len(dir_borderline_list))
        
    except:
        print("video dataset directory is not valid")

    # augmentation argument
    augment_arg = {
        "bright_val" : args['bright_val'],
        "bright_p" : args['bright_p'],
        "contrast_min" : args['contrast_min'],
        "contrast_max" : args['contrast_max'],
        "contrast_p" : args['contrast_p'],
        "blur_k" : args['blur_k'],
        "blur_p" : args['blur_p'],
        "flip_p" : args['flip_p'],
        "vertical_ratio" : args['vertical_ratio'],
        "vertical_p" : args['vertical_p'],
        "horizontal_ratio" : args['horizontal_ratio'],
        "horizontal_p" : args['horizontal_p']
    }
    
    train_data = VideoDataset(dataset = args["dataset"], split = "train", clip_len = args["clip_len"], preprocess = False, augmentation = True, augmentation_args=augment_arg)
    valid_data = VideoDataset(dataset = args["dataset"], split = "val", clip_len = args["clip_len"], preprocess = False, augmentation=False)
    test_data = VideoDataset(dataset = args["dataset"], split = "test", clip_len = args["clip_len"], preprocess = False, augmentation=False)

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

    # define input shape
    input_shape = (args["channel"], args['clip_len'], args['height'], args['width'])

    # define model
    model = None
    
    print("\n################# model summary #################\n")
    model.summary()
    model.to(device)

    # optimizer
    if args["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "RMSProps":
        optimizer = torch.optim.RMSprop(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])

    # conventional scheduler
    if args["use_scheduler"] and args['train_rule'] == "None":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = args["T_0"],
            T_mult = args["T_mult"]
        )
    elif args['train_rule'] == 'DRW':
        scheduler = "DRW"
    else:
        scheduler = None

    # train rule
    if args['train_rule'] == 'None':
        train_sampler = None
        per_cls_weights = None
    elif args['train_rule'] == 'Resample':
        train_sampler = ImbalancedDatasetSampler(train_data)
        per_cls_weights = None
    elif args['train_rule'] == 'Reweight':
        cls_num_list = train_data.get_cls_num_list()
        train_sampler = None
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)

    elif args['train_rule'] == 'DRW':
        # we have to update beta [0, 0.9999] by changing idx with respect to epochs
        train_sampler = None
        idx = 0
        beta = 0
        cls_num_list = train_data.get_cls_num_list()
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    else:
        train_sampler = None
        per_cls_weights = None

    # loss
    if args['loss_type'] == "CE":
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")
    elif args['loss_type'] == 'LDAM':
        loss_fn = LDAMLoss(cls_num_list, max_m = 0.5, s = 30, weight = per_cls_weights)
    elif args['loss_type'] == 'Focal':
        loss_fn = FocalLoss(per_cls_weights, gamma = 1.0)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")

    # training process
    print("\n################# training process #################\n")

    train_loss,  train_acc, valid_loss, valid_acc = train_(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        args['num_epoch'],
        args['verbose'],
        save_best = args['save_best'],
        use_video_mixup = args['use_video_mixup'],
        use_acc_per_class = False,
        train_rule = args['train_rule'],
        cls_num_list = cls_num_list
    )
   
    print("\n################# evaluation process #################\n")
    
    model.load_state_dict(torch.load(args["save_best"]))

    test_loss, test_acc = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
        save_dir = args["save_evaluation"]
    )
    
    x_axis = range(1, args['num_epoch'] + 1)
    plt.figure(figsize = (16,10))
    plt.subplot(1,2,1)
    plt.plot(x_axis, train_loss, 'ro-', label  = "train loss")
    plt.plot(x_axis, valid_loss, 'b^-', label = "valid loss")
    plt.xlabel("epoch")
    plt.ylabel("loss : ", args['loss_type'])
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(x_axis, train_acc, 'ro--', label = "train acc")
    plt.plot(x_axis, valid_acc, 'bo--', label = 'valid acc')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(args["save_training_curve"])