import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.dataset import DatasetForVideo
from torch.utils.data import DataLoader
from src.utils.utility import preparing_video_dataset, plot_learning_curve, generate_prob_curve, seed_everything
from src.visualization.visualize_latent_space import visualize_2D_latent_space, visualize_3D_latent_space
from src.evaluate import evaluate, evaluate_detail
from src.loss import FocalLoss, LDAMLoss, CELoss
from src.models.ViViT import ViViT
from src.models.R2Plus1D import R2Plus1DClassifier
from src.models.resnet import Bottleneck3D
from src.models.slowfast import SlowFast

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Experiment for ViViT model")
    
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
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--seq_len", type = int, default = 21)
    parser.add_argument("--dist", type = int, default = 3)
    parser.add_argument("--num_workers", type = int, default = 8)
    parser.add_argument("--pin_memory", type = bool, default = False)
    
    # Re-sampling
    parser.add_argument("--use_sampling", type = bool, default = False)
    
    # Re-weighting
    parser.add_argument("--use_weighting", type = bool, default = False)
    
    # Deffered Re-weighting
    parser.add_argument("--use_DRW", type = bool, default = False)

    # loss type : CE, Focal, LDAM
    parser.add_argument("--loss_type", type = str, default = "Focal", choices = ['CE','Focal', 'LDAM'])

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
    
    # save directory
    save_dir = args['save_dir']
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    if args['model'] == 'SlowFast' and args['seq_len'] % 2 == 1:
        print("SlowFast : seq_len must be even number, seq_len-1 as input")
        args['seq_len'] -= 1
        
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
    
    tag = "{}_clip_{}_dist_{}_{}_{}_seed_{}".format(args["tag"], args["seq_len"], args["dist"], loss_type, boost_type, args['random_seed'])
    
    print("================= Running code =================")
    print("Setting : {}".format(tag))

    save_best_dir = "./weights/{}_best.pt".format(tag)
    save_last_dir = "./weights/{}_last.pt".format(tag)
 
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'

    # use modified dataset
    root_dir = "./dataset/temp"
    shot_train, shot_valid, shot_test = preparing_video_dataset(root_dir,test_shot = args['test_shot_num'])
    df_disrupt = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_extend.csv")
    
    train_data = DatasetForVideo(shot_train, df_disrupt, augmentation = False, augmentation_args=None, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])
    valid_data = DatasetForVideo(shot_valid, df_disrupt, augmentation = False, augmentation_args=None, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])
    test_data = DatasetForVideo(shot_test, df_disrupt, augmentation = False, augmentation_args=None, crop_size = args['image_size'], seq_len = args['seq_len'], dist = args['dist'])
    
    print("================= Dataset information =================")
    print("train data : {}, disrupt : {}, non-disrupt : {}".format(train_data.__len__(), train_data.n_disrupt, train_data.n_normal))
    print("valid data : {}, disrupt : {}, non-disrupt : {}".format(valid_data.__len__(), valid_data.n_disrupt, valid_data.n_normal))
    print("test data : {}, disrupt : {}, non-disrupt : {}".format(test_data.__len__(), test_data.n_disrupt, test_data.n_normal))
    
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
        
    print("\n==================== model summary ====================\n")
    model.summary(show_hierarchical = False, show_parent_layers=True)
    model.to(device)
    
    train_loader = DataLoader(train_data, batch_size = args['batch_size'], sampler=None, num_workers = args["num_workers"], pin_memory=args["pin_memory"], drop_last = True)
    valid_loader = DataLoader(valid_data, batch_size = args['batch_size'], sampler=None, num_workers = args["num_workers"], pin_memory=args["pin_memory"], drop_last = True)
    test_loader = DataLoader(test_data, batch_size = args['batch_size'], sampler=None, num_workers = args["num_workers"], pin_memory=args["pin_memory"], drop_last = True)

    # load best weight
    model.load_state_dict(torch.load(save_best_dir))
    
    save_conf = os.path.join(save_dir, "{}_test_confusion.png".format(tag))
    save_txt = os.path.join(save_dir, "{}_test_eval.txt".format(tag))
    
    # Additional analyzation
    print("\n====================== Visualization process ======================\n")
    try:
        visualize_2D_latent_space(
            model, 
            train_loader,
            device,
            os.path.join(save_dir, "{}_2D_latent_train.png".format(tag)),
            3,
            'tSNE'
        )
        
        visualize_2D_latent_space(
            model, 
            test_loader,
            device,
            os.path.join(save_dir, "{}_2D_latent_test.png".format(tag)),
            3,
            'tSNE'
        )
        
    except:
        print("{} : visualize 2D latent space doesn't work due to stability error".format(tag))
    
    try:
        visualize_3D_latent_space(
            model, 
            train_loader,
            device,
            os.path.join(save_dir, "{}_3D_latent_train.png".format(tag)),
            3,
            'tSNE'
        )
        
        visualize_3D_latent_space(
            model, 
            test_loader,
            device,
            os.path.join(save_dir, "{}_3D_latent_test.png".format(tag)),
            3,
            'tSNE'
        )
        
    except:
        print("{} : visualize 3D latent space doesn't work due to stability error".format(tag))
    
    # plot the disruption probability curve
    test_shot_num = args['test_shot_num']
    print("\n====================== Probability curve generation process ======================\n")
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
    
    print("\n====================== Detail evaluation for each experiment ======================\n")
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