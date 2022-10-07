import torch
import pandas as pd
import numpy as np
import argparse
import copy
from torch.utils.data import DataLoader
from src.CustomDataset import DEFAULT_TS_COLS, MultiModalDataset
from src.utils.sampler import ImbalancedDatasetSampler
from src.train import train, train_DRW
from src.evaluate import evaluate
from src.loss import LDAMLoss, FocalLoss
from src.visualization.visualize_latent_space import visualize_3D_latent_space_multi
from src.GradientBlending import GradientBlending, train_GB_dynamic, train_GB
from src.CCA import DeepCCA, train_cca, CCALoss, evaluate_cca_loss
from src.models.MultiModal import TFN, TFN_GB

parser = argparse.ArgumentParser(description="training multimodal network for disruption classifier")
parser.add_argument("--batch_size", type = int, default = 32)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--step_size", type = int, default = 4)
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--tag", type = str, default = '')

parser.add_argument("--num_workers", type = int, default = 8)
parser.add_argument("--pin_memory", type = bool, default = False)

parser.add_argument("--use_sampler", type = bool, default = True)
parser.add_argument("--num_epoch", type = int, default = 64)
parser.add_argument("--verbose", type = int, default = 8)
parser.add_argument("--dist", type = int, default = 4)
parser.add_argument("--seq_len", type = int, default = 21)

parser.add_argument("--use_focal_loss", type = bool, default = True)
parser.add_argument("--use_LDAM_loss", type = bool, default = False)
parser.add_argument("--use_weight", type = bool, default = True)
parser.add_argument("--use_DRW", type = bool, default = False)
parser.add_argument("--use_GB", type = bool, default = False)
parser.add_argument("--use_CCA", type = bool, default = False)

args = vars(parser.parse_args())

# default argument
args_video = {
    "image_size" : 128, 
    "patch_size" : 16, 
    "n_frames" : 21, 
    "dim": 64, 
    "depth" : 4, 
    "n_heads" : 8, 
    "pool" : 'cls', 
    "in_channels" : 3, 
    "d_head" : 64, 
    "dropout" : 0.25,
    "embedd_dropout":  0.25, 
    "scale_dim" : 4,
}

args_0D = {
    "seq_len" : 21, 
    "col_dim" : 9, 
    "conv_dim" : 32, 
    "conv_kernel" : 3,
    "conv_stride" : 1, 
    "conv_padding" : 1,
    "lstm_dim" : 32, 
}

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

if __name__ == "__main__":
    
    # directory setting
    dist = args['dist']
    seq_len = args['seq_len']
    tag = args['tag']
    args['root_dir'] = "./dataset/dur{}_dis4{}".format(seq_len * 4, dist)
    args['save_best_dir'] = "./weights/multi_modal_{}_clip_{}_dist_{}_best.pt".format(tag,seq_len, dist)
    args['save_last_dir'] = "./weights/multi_modal_{}_clip_{}_dist_{}_last.pt".format(tag, seq_len, dist)
    args['save_result_dir'] = "./results/loss_curve_multi_modal_{}_clip_{}_dist_{}.png".format(tag, seq_len, dist)
    args['save_txt'] = "./results/test_multi_modal_{}_clip_{}_dist_{}.txt".format(tag, seq_len, dist)
    args['save_conf'] = "./results/test_multi_modal_{}_clip_{}_dist_{}_confusion_matrix.png".format(tag, seq_len, dist)
    args['save_latent_dir'] = "./results/test_multi_modal_{}_clip_{}_dist_{}_3d_latent.png".format(tag, seq_len, dist)  
    
    kstar_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_extend.csv', encoding = "euc-kr")
    ts_data = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_for_multi.csv")
    mult_info = pd.read_csv("./dataset/KSTAR_Disruption_multi_data.csv")

    train_data = MultiModalDataset('train', ts_data, DEFAULT_TS_COLS, mult_info, dt = 1 / 210 * 4, distance = args['dist'], seq_len = args['seq_len'])
    valid_data = MultiModalDataset('valid', ts_data, DEFAULT_TS_COLS, mult_info, dt = 1 / 210 * 4, distance = args['dist'], seq_len = args['seq_len'])
    test_data = MultiModalDataset('test', ts_data, DEFAULT_TS_COLS, mult_info, dt = 1 / 210 * 4, distance = args['dist'], seq_len = args['seq_len'])

    batch_size = args["batch_size"]
    lr = args['lr']
    step_size = args['step_size']

    # Re-Sampling
    if args["use_sampler"]:
        train_sampler = ImbalancedDatasetSampler(train_data)
        valid_sampler = None
        test_sampler = None

    else:
        train_sampler = None
        valid_sampler = None
        test_sampler = None
        
    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers = args['num_workers'], sampler = train_sampler)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = args['num_workers'], sampler = valid_sampler)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = args['num_workers'], sampler = test_sampler)

    train_data.get_num_per_cls()
    cls_num_list = train_data.get_cls_num_list()
    
    # Re-Weighting
    if args['use_weight']:
        per_cls_weights = 1.0 / np.array(cls_num_list)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    else:
        per_cls_weights = np.array([1,1])
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        
    # use Deferred Re-Weighting
    if args['use_DRW']:
        betas = [0, 0.25, 0.5, 0.75]
    else:
        betas = None
    
    # Loss function : Focal / LDAM / CE
    if args['use_focal_loss']:
        focal_gamma = 2.0
        loss_fn = FocalLoss(weight = per_cls_weights, gamma = focal_gamma)
    elif args['use_LDAM_loss']:
        max_m = 0.5
        s = 1.0
        loss_fn = LDAMLoss(cls_num_list, max_m = max_m, weight = per_cls_weights, s = s)
    else: 
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean", weight = per_cls_weights)
        
    # Gradient Blending
    if args['use_GB']:
        w_fusion = 0.5
        w_vis = 0.1
        w_0D = 0.4
        loss_fn = GradientBlending(
            copy.deepcopy(loss_fn),
            copy.deepcopy(loss_fn),
            copy.deepcopy(loss_fn),
            w_vis,
            w_0D,
            w_fusion   
        )
        
    # model 
    if args['use_GB']:
        model = TFN_GB(
            2, 128, args_video, args_0D
        )
    else:
        model = TFN(
            2, 128, args_video, args_0D
        )
    
    model.summary('cpu')
    model.to(device)
    
    # CCA Learning
    if args['use_CCA']:
        deep_cca = DeepCCA(
            model.network_video,
            model.network_0D,
        )
        
        cca_loss = CCALoss(
            output_dim = model.encoder_dims,
            use_all_singular_values=True
        )
        
        optim_deep = torch.optim.RMSprop(deep_cca.parameters(), lr = lr)
        sched_deep = torch.optim.lr_scheduler.StepLR(optim_deep, step_size = step_size, gamma=args['gamma'])
        
        train_cca(
            train_loader,
            valid_loader,
            deep_cca,
            optim_deep,
            sched_deep,
            cca_loss,
            device,
            32,
            4,
            "./weights/cca_best.pt",
            "./weights/cca_last.pt",
            0.25
        )
        
        test_cca_loss = evaluate_cca_loss(
            test_loader,
            model,
            deep_cca,
            device
        )
        
        deep_cca.cpu()
        
    # training for multimodal disruption prediction task
    model.to(device)
    num_epoch = args['num_epoch']
    verbose = args['verbose']
    max_norm_grad = 1.0
    criteria = "f1_score"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma=args['gamma'])

    if args['use_GB']:
        train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train_GB(
            train_loader,
            valid_loader,
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            num_epoch,
            verbose,
            args['save_best_dir'],
            args['save_last_dir'],
            max_norm_grad,
            criteria,
        )
        
        model.load_state_dict(torch.load(args['save_best_dir']))

        # evaluation process
        test_loss, test_acc, test_f1 = evaluate(
            test_loader,
            model,
            optimizer,
            loss_fn,
            device,
            save_conf = args['save_conf'],
            save_txt = args['save_txt'],
            model_type = 'multi-GB'
        )
    elif args['use_DRW']:
        train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train_DRW(
            train_loader,
            valid_loader,
            model,
            optimizer,
            loss_fn,
            device,
            num_epoch,
            verbose,
            args['save_best_dir'],
            args['save_last_dir'],
            max_norm_grad,
            criteria,
            cls_num_list,
            betas,
            "multi"
        )
    else:
        train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train(
            train_loader,
            valid_loader,
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            num_epoch,
            verbose,
            args['save_best_dir'],
            args['save_last_dir'],
            max_norm_grad,
            criteria,
            "multi"
        )
        
        model.load_state_dict(torch.load(args['save_best_dir']))
        
        # evaluation process
        test_loss, test_acc, test_f1 = evaluate(
            test_loader,
            model,
            optimizer,
            loss_fn,
            device,
            save_conf = args['save_conf'],
            save_txt = args['save_txt'],
            model_type = 'multi'
        )

    # plot the 3d latent space
    visualize_3D_latent_space_multi(
        model, 
        train_loader,
        device,
        args["save_latent_dir"], 
    )
    
    # plot attention weight