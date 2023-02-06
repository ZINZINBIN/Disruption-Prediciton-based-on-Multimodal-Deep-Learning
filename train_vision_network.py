import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.dataset import DatasetForVideo2
from torch.utils.data import DataLoader, RandomSampler
from src.utils.sampler import ImbalancedDatasetSampler
from src.utils.utility import preparing_video_dataset, plot_learning_curve, generate_prob_curve, seed_everything
from src.visualization.visualize_latent_space import visualize_2D_latent_space, visualize_3D_latent_space
from src.train import train, train_DRW
from src.evaluate import evaluate
from src.loss import FocalLoss, LDAMLoss, CELoss
from src.models.ViViT import ViViT
from src.models.R2Plus1D import R2Plus1DClassifier
from src.models.resnet import Bottleneck3D
from src.models.slowfast import SlowFast

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="training disruption prediction model with video data")
    
    # random seed
    parser.add_argument("--random_seed", type = int, default = 42)
    
    # tag and result directory
    parser.add_argument("--model", type = str, default = 'ViViT', choices=['ViViT', 'SlowFast', 'R2Plus1D'])
    parser.add_argument("--tag", type = str, default = "ViViT")
    parser.add_argument("--save_dir", type = str, default = "./results")
    
    # test shot for disruption probability curve
    parser.add_argument("--test_shot_num", type = int, default = 21310)

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)

    # data input shape 
    parser.add_argument("--image_size", type = int, default = 128)

    # common argument
    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--num_epoch", type = int, default = 128)
    parser.add_argument("--seq_len", type = int, default = 21)
    parser.add_argument("--dist", type = int, default = 3)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = False)

    # model weight / save process
    # wandb setting
    parser.add_argument("--use_wandb", type = bool, default = False)
    parser.add_argument("--wandb_save_name", type = str, default = "SBERT-exp001")
    
    # detail setting for training process
    # data augmentation : conventional
    parser.add_argument("--bright_val", type = int, default = 10)
    parser.add_argument("--bright_p", type = float, default = 0.25)
    parser.add_argument("--contrast_min", type = float, default = 1)
    parser.add_argument("--contrast_max", type = float, default = 1.25)
    parser.add_argument("--contrast_p", type = float, default = 0.25)
    parser.add_argument("--blur_k", type = int, default = 5)
    parser.add_argument("--blur_p", type = float, default = 0.25)
    parser.add_argument("--flip_p", type = float, default = 0.25)
    parser.add_argument("--vertical_ratio", type = float, default = 0.1)
    parser.add_argument("--vertical_p", type = float, default = 0.25)
    parser.add_argument("--horizontal_ratio", type = float, default = 0.1)
    parser.add_argument("--horizontal_p", type = float, default = 0.25)
    
    # optimizer : SGD, RMSProps, Adam, AdamW
    parser.add_argument("--optimizer", type = str, default = "AdamW")
    
    # learning rate, step size and decay constant
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--use_scheduler", type = bool, default = True)
    parser.add_argument("--step_size", type = int, default = 4)
    parser.add_argument("--gamma", type = float, default = 0.95)
    
    # early stopping
    parser.add_argument('--early_stopping', type = bool, default = True)
    parser.add_argument("--early_stopping_patience", type = int, default = 12)
    parser.add_argument("--early_stopping_verbose", type = bool, default = True)
    parser.add_argument("--early_stopping_delta", type = float, default = 1e-3)
    
    # imbalanced dataset processing
    # Re-sampling
    parser.add_argument("--use_sampling", type = bool, default = True)
    
    # Re-weighting
    parser.add_argument("--use_weighting", type = bool, default = True)
    
    # Deffered Re-weighting
    parser.add_argument("--use_DRW", type = bool, default = False)
    parser.add_argument("--beta", type = float, default = 0.25)

    # loss type : CE, Focal, LDAM
    parser.add_argument("--loss_type", type = str, default = "Focal", choices = ['CE','Focal', 'LDAM'])
    
    # LDAM Loss parameter
    parser.add_argument("--max_m", type = float, default = 0.5)
    parser.add_argument("--s", type = float, default = 1.0)
    
    # Focal Loss parameter
    parser.add_argument("--focal_gamma", type = float, default = 2.0)
    
    # monitoring the training process
    parser.add_argument("--verbose", type = int, default = 4)
    
    # model setup
    # ViViT
    parser.add_argument("--patch_size", type = int, default = 16)
    parser.add_argument("--alpha", type = float, default = 1.0)
    parser.add_argument("--dropout", type = float, default = 0.1)
    parser.add_argument("--embedd_dropout", type = float, default = 0.1)
    parser.add_argument("--dim", type = int, default = 128)
    parser.add_argument("--n_heads", type = int, default = 4)
    parser.add_argument("--d_head", type = int, default = 64)
    parser.add_argument("--scale_dim", type = int, default = 8)
    parser.add_argument("--depth", type = int, default = 2)
    
    # SlowFast
    parser.add_argument("--tau_alpha", type = int, default = 4)
    parser.add_argument("--tau_fast", type = int, default = 1)
    
    # R2Plus1D + SlowFast
    parser.add_argument("--n_layer", type = int, default = 2)
    
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
    
    # seed initialize
    seed_everything(args['random_seed'], False)
    
    # save directory
    save_dir = args['save_dir']
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    if args['model'] == 'SlowFast' and args['seq_len'] % 2 == 1:
        print("SlowFast : seq_len must be even number, seq_len-1 as input")
        args['seq_len'] -= 1
    
    # tag : {model_name}_clip_{seq_len}_dist_{pred_len}_{Loss-type}_{Boosting-type}
    loss_type = args['loss_type']
    
    if args['use_sampling'] and not args['use_weighting'] and not args['use_DRW']:
        boost_type = "RS"
    elif args['use_sampling'] and args['use_weighting'] and not args['use_DRW']:
        boost_type = "RS_RW"
    elif args['use_sampling'] and not args['use_weighting'] and args['use_DRW']:
        boost_type = "RS_DRW"
    elif args['use_sampling'] and args['use_weighting'] and args['use_DRW']:
        boost_type = "RS_DRW"
    elif not args['use_sampling'] and args['use_weighting'] and not args['use_DRW']:
        boost_type = "RW"
    elif not args['use_sampling'] and not args['use_weighting'] and args['use_DRW']:
        boost_type = "DRW"
    elif not args['use_sampling'] and args['use_weighting'] and args['use_DRW']:
        boost_type = "DRW"
    elif not args['use_sampling'] and not args['use_weighting'] and not args['use_DRW']:
        boost_type = "Normal"
    
    tag = "{}_clip_{}_dist_{}_{}_{}".format(args["tag"], args["seq_len"], args["dist"], loss_type, boost_type)
    
    print("running : {}".format(tag))

    save_best_dir = "./weights/{}_best.pt".format(tag)
    save_last_dir = "./weights/{}_last.pt".format(tag)
    exp_dir = os.path.join("./runs/", "tensorboard_{}".format(tag))
 
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'

    # augmentation argument
    augment_args = {
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
    
    # use modified dataset
    root_dir = "./dataset/temp"
    shot_train, shot_valid, shot_test = preparing_video_dataset(root_dir)
    df_disrupt = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_extend.csv")
    
    train_data = DatasetForVideo2(shot_train, df_disrupt, augmentation = True, augmentation_args=augment_args, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])
    valid_data = DatasetForVideo2(shot_valid, df_disrupt, augmentation = False, augmentation_args=augment_args, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])
    test_data = DatasetForVideo2(shot_test, df_disrupt, augmentation = False, augmentation_args=augment_args, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])
    
    print("train data : {}, disrupt : {}, non-disrupt : {}".format(train_data.__len__(), train_data.n_disrupt, train_data.n_normal))
    print("valid data : {}, disrupt : {}, non-disrupt : {}".format(valid_data.__len__(), valid_data.n_disrupt, valid_data.n_normal))
    print("test data : {}, disrupt : {}, non-disrupt : {}".format(test_data.__len__(), test_data.n_disrupt, test_data.n_normal))
    
    # label distribution for LDAM / Focal Loss
    train_data.get_num_per_cls()
    cls_num_list = train_data.get_cls_num_list()

    # define model
    if args['model'] == 'ViViT':
        model = ViViT(
            image_size = args['image_size'],
            patch_size = args['patch_size'],
            n_classes = 2,
            n_frames = args['seq_len'],
            dim = args['dim'],
            depth = args['depth'],
            n_heads = args['n_heads'],
            pool = "mean",
            in_channels = 3,
            d_head = args['d_head'],
            dropout = args['dropout'],
            embedd_dropout=args['embedd_dropout'],
            scale_dim = args['scale_dim'],
            alpha = args['alpha']
        )
        
    elif args['model'] == 'SlowFast':
                   
        model = SlowFast(
            input_shape = (3, args['seq_len'], args['image_size'], args['image_size']),
            block = Bottleneck3D,
            layers = [1,args['n_layer'],args['n_layer'],1],
            alpha = args['tau_alpha'],
            tau_fast = args['tau_fast'],
            num_classes = 2,
            alpha_elu = args['alpha'],
        )
        
    elif args['model'] == 'R2Plus1D':
        model = R2Plus1DClassifier(
            input_size  = (3, args['seq_len'], args['image_size'], args['image_size']),
            num_classes = 2, 
            layer_sizes = [1,args['n_layer'],args['n_layer'],1],
            pretrained = False, 
            alpha = args['alpha']
        )
        
    
    print("\n################# model summary #################\n")
    model.summary(show_hierarchical = False, show_parent_layers=True)
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
        
    # scheduler
    if args["use_scheduler"] and not args["use_DRW"]:    
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args['step_size'], gamma=args['gamma'])
        
    elif args["use_DRW"]:
        scheduler = "DRW"
        
    else:
        scheduler = None
        
    # Re-sampling
    if args["use_sampling"]:
        train_sampler = ImbalancedDatasetSampler(train_data)
        valid_sampler = RandomSampler(valid_data)
        test_sampler = RandomSampler(test_data)

    else:
        train_sampler = RandomSampler(train_data)
        valid_sampler = RandomSampler(valid_data)
        test_sampler = RandomSampler(test_data)
    
    train_loader = DataLoader(train_data, batch_size = args['batch_size'], sampler=train_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    valid_loader = DataLoader(valid_data, batch_size = args['batch_size'], sampler=valid_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    test_loader = DataLoader(test_data, batch_size = args['batch_size'], sampler=test_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])

    # Re-weighting
    if args['use_weighting']:
        per_cls_weights = 1.0 / np.array(cls_num_list)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    else:
        per_cls_weights = np.array([1,1])
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        
    # loss
    if args['loss_type'] == "CE":
        betas = [0, args['beta'], args['beta'] * 2, args['beta']*3]
        loss_fn = CELoss(weight = per_cls_weights)
    elif args['loss_type'] == 'LDAM':
        max_m = args['max_m']
        s = args['s']
        betas = [0, args['beta'], args['beta'] * 2, args['beta']*3]
        loss_fn = LDAMLoss(cls_num_list, max_m = max_m, s = s, weight = per_cls_weights)
    elif args['loss_type'] == 'Focal':
        betas = [0, args['beta'], args['beta'] * 2, args['beta']*3]
        focal_gamma = args['focal_gamma']
        loss_fn = FocalLoss(weight = per_cls_weights, gamma = focal_gamma)
    else:
        betas = [0, args['beta'], args['beta'] * 2, args['beta']*3]
        loss_fn = CELoss(weight = per_cls_weights)

    # training process
    print("\n################# training process #################\n")
    if args['use_DRW']:
        train_loss,  train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train_DRW(
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
            exp_dir = exp_dir,
            max_norm_grad = 1.0,
            cls_num_list=cls_num_list,
            betas = betas,
            model_type = "single",
            test_for_check_per_epoch=test_loader,
            is_early_stopping = args['early_stopping'],
            early_stopping_verbose = args['early_stopping_verbose'],
            early_stopping_patience = args['early_stopping_patience'],
            early_stopping_delta = args['early_stopping_delta']
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
            exp_dir = exp_dir,
            max_norm_grad = 1.0,
            model_type = "single",
            test_for_check_per_epoch=test_loader,
            is_early_stopping = args['early_stopping'],
            early_stopping_verbose = args['early_stopping_verbose'],
            early_stopping_patience = args['early_stopping_patience'],
            early_stopping_delta = args['early_stopping_delta']
        )
    
    
    # plot the learning curve
    save_learning_curve = os.path.join(save_dir, "{}_lr_curve.png".format(tag))
    plot_learning_curve(train_loss, valid_loss, train_f1, valid_f1, figsize = (12,6), save_dir = save_learning_curve)
    
    # evaluation process
    print("\n################# evaluation process #################\n")
    model.load_state_dict(torch.load(save_best_dir))
    
    save_conf = os.path.join(save_dir, "{}_test_confusion.png".format(tag))
    save_txt = os.path.join(save_dir, "{}_test_eval.txt".format(tag))
    
    test_loss, test_acc, test_f1 = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
        save_conf = save_conf,
        save_txt = save_txt,
    )
    
    # Additional analyzation
    save_2D_latent_dir = os.path.join(save_dir, "{}_2D_latent.png".format(tag))
    save_3D_latent_dir = os.path.join(save_dir, "{}_3D_latent.png".format(tag))
    print("\n################# Visualization process #################\n")
    try:
        visualize_2D_latent_space(
            model, 
            train_loader,
            device,
            save_2D_latent_dir
        )
        
    except:
        print("{} : visualize 2D latent space doesn't work due to stability error".format(tag))
    
    try:
        visualize_3D_latent_space(
            model, 
            train_loader,
            device,
            save_3D_latent_dir
        )
    except:
        print("{} : visualize 3D latent space doesn't work due to stability error".format(tag))
        
    # plot the disruption probability curve
    test_shot_num = args['test_shot_num']
    print("\n################# Probability curve generation process #################\n")
    time_x, prob_list = generate_prob_curve(
        file_path = "./dataset/temp/{}".format(test_shot_num),
        model = model, 
        device = device, 
        save_dir = os.path.join(save_dir, "{}_probs_curve_{}.png".format(tag, test_shot_num)),
        shot_list_dir = "./dataset/KSTAR_Disruption_Shot_List_extend.csv",
        ts_data_dir = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
        shot_num = test_shot_num,
        clip_len = args['seq_len'],
        dist_frame = args['dist'],
    )