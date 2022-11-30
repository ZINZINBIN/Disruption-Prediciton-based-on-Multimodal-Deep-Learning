# gif file generation
import torch
import os
import numpy as np
from src.models.ViViT import ViViT
from src.models.ts_transformer import TStransformer
from src.visualization.visualize_application import generate_real_time_experiment, video2img, VideoDataset, generate_real_time_experiment_0D

ts_cols = [
    '\\q95', '\\ipmhd', '\\kappa', 
    '\\tritop', '\\tribot','\\betap','\\betan',
    '\\li', '\\WTOT_DLM03'
]

args = {
    "seq_len" : 21,
    "vis_save_best_dir" : "./weights/ViViT_clip_21_dist_3_best.pt",
    # "0D_save_best_dir" : "./weights/ts_conv_lstm_clip_21_dist_3_best.pt",
    "0D_save_best_dir" : "./weights/ts_transformer_clip_21_dist_3_best.pt",
    "image_size" : 128,
    "patch_size" : 16,
    "dist" : 3,
    "gpu_num" : 0,
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
    
    model = ViViT(
        image_size = args['image_size'],
        patch_size = args['patch_size'],
        n_classes = 2,
        n_frames = args['seq_len'],
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
    model.load_state_dict(torch.load(args['vis_save_best_dir']))
    
    shot_num = 21310
    file_path = os.path.join("./dataset/raw_videos/raw_videos/", "%06dtv01.avi"%shot_num)
    img_file_path = os.path.join("./dataset/test-shot")
    # video2img(file_path, 256, 256, True, img_file_path)
    
    # dataset = VideoDataset(img_file_path, resize_height = 256, resize_width=256, crop_size = 128, seq_len = 21, dist = 3, frame_srt=21, frame_end = -1)
    
    # generate_real_time_experiment(
    #     img_file_path,
    #     model, 
    #     device, 
    #     save_dir = "./results/real_time_disruption_prediction_{}.gif".format(shot_num),
    #     shot_list_dir = "./dataset/KSTAR_Disruption_Shot_List_extend.csv",
    #     ts_data_dir = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
    #     shot_num = shot_num,
    #     clip_len = args['seq_len'],
    #     dist_frame = args['dist'],
    #     plot_freq = 7,
    # )
    
    model.cpu()
    
    del model
    
    # 0D data
    model = TStransformer(
        n_features=len(ts_cols),
        feature_dims = 128,
        max_len = args['seq_len'],
        n_layers = 8,
        n_heads = 8,
        dim_feedforward=1024,
        dropout = 0.25,
        cls_dims = 128,
        n_classes = 2
    )
    
    model.to(device)
    model.load_state_dict(torch.load(args['0D_save_best_dir']))
    
    generate_real_time_experiment_0D(
        img_file_path,
        model, 
        device, 
        save_dir = "./results/real_time_disruption_prediction_0D_{}.gif".format(shot_num),
        shot_list_dir = "./dataset/KSTAR_Disruption_Shot_List_extend.csv",
        ts_data_dir = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
        shot_num = shot_num,
        clip_len = args['seq_len'],
        dist_frame = args['dist'],
        plot_freq = 7,
    )