import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.CustomDataset import DatasetForVideo2
from torch.utils.data import DataLoader
from src.utils.utility import preparing_video_dataset
from src.visualization.visualize_latent_space import visualize_2D_latent_space, visualize_3D_latent_space
from src.evaluate import evaluate_detail
from src.models.ViViT import ViViT

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Experiment for ViViT model")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "ViViT")
    parser.add_argument("--save_dir", type = str, default = "./results")

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)

    # data input shape 
    parser.add_argument("--image_size", type = int, default = 256)
    parser.add_argument("--patch_size", type = int, default = 16)

    # common argument
    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--seq_len", type = int, default = 21)
    parser.add_argument("--dist", type = int, default = 3)
    parser.add_argument("--num_workers", type = int, default = 8)
    parser.add_argument("--pin_memory", type = bool, default = False)

    # model setup
    parser.add_argument("--alpha", type = float, default = 0.01)
    parser.add_argument("--dropout", type = float, default = 0.25)
    parser.add_argument("--embedd_dropout", type = float, default = 0.25)
    parser.add_argument("--dim", type = int, default = 128)
    parser.add_argument("--n_heads", type = int, default = 8)
    parser.add_argument("--d_head", type = int, default = 64)
    parser.add_argument("--scale_dim", type = int, default = 4)
    parser.add_argument("--depth", type = int, default = 4)
    
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
    
    # save directory
    save_dir = args['save_dir']
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    tag = "{}_clip_{}_dist_{}".format(args["tag"], args["seq_len"], args["dist"])
    save_best_dir = "./weights/{}_best.pt".format(tag)
    save_last_dir = "./weights/{}_last.pt".format(tag)
 
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'

    # use modified dataset
    root_dir = "./dataset/temp"
    shot_train, shot_valid, shot_test = preparing_video_dataset(root_dir)
    df_disrupt = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_extend.csv")
    
    train_data = DatasetForVideo2(shot_train, df_disrupt, augmentation = False, augmentation_args=None, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])
    valid_data = DatasetForVideo2(shot_valid, df_disrupt, augmentation = False, augmentation_args=None, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])
    test_data = DatasetForVideo2(shot_test, df_disrupt, augmentation = False, augmentation_args=None, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])
    
    print("train data : {}, disrupt : {}, non-disrupt : {}".format(train_data.__len__(), train_data.n_disrupt, train_data.n_normal))
    print("valid data : {}, disrupt : {}, non-disrupt : {}".format(valid_data.__len__(), valid_data.n_disrupt, valid_data.n_normal))
    print("test data : {}, disrupt : {}, non-disrupt : {}".format(test_data.__len__(), test_data.n_disrupt, test_data.n_normal))
    
    # define model
    model = ViViT(
        image_size = args['image_size'],
        patch_size = args['patch_size'],
        n_classes = 2,
        n_frames = args['seq_len'],
        dim = args['dim'],
        depth = args['depth'],
        n_heads = args['n_heads'],
        pool = "cls",
        in_channels = 3,
        d_head = args['d_head'],
        dropout = args['dropout'],
        embedd_dropout=args['embedd_dropout'],
        scale_dim = args['scale_dim']
    )
    
    model.to(device)
    
    train_loader = DataLoader(train_data, batch_size = args['batch_size'], sampler=None, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    valid_loader = DataLoader(valid_data, batch_size = args['batch_size'], sampler=None, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    test_loader = DataLoader(test_data, batch_size = args['batch_size'], sampler=None, num_workers = args["num_workers"], pin_memory=args["pin_memory"])

    model.load_state_dict(torch.load(save_best_dir))
    
    save_csv = os.path.join(save_dir, "{}_total_score.csv".format(tag))
    
    evaluate_detail(
        train_loader,
        valid_loader,
        test_loader,
        model,
        device,
        save_csv,
        tag,
        model_type = 'single'
    )