import torch
import argparse
import numpy as np
import pandas as pd
from src.CustomDataset import DatasetFor0D
from src.models.ts_transformer import TStransformer
from src.utils.sampler import ImbalancedDatasetSampler
from src.utils.utility import plot_learning_curve, generate_prob_curve_from_0D
from src.visualization.visualize_latent_space import visualize_2D_latent_space
from torch.utils.data import DataLoader
from src.train import train
from src.evaluate import evaluate
from src.loss import LDAMLoss, FocalLoss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index()

# nan interpolation
df.interpolate(method = 'linear', limit_direction = 'forward')

# columns for use
ts_cols = [
    '\\q95', '\\ipmhd', '\\kappa', 
    '\\tritop', '\\tribot','\\betap','\\betan',
    '\\li', '\\WTOT_DLM03'
]

# float type
for col in ts_cols:
    df[col] = df[col].astype(np.float32)

# train / valid / test data split
from sklearn.model_selection import train_test_split
shot_list = np.unique(df.shot.values)

shot_train, shot_test = train_test_split(shot_list, test_size = 0.2, random_state = 42)
shot_train, shot_valid = train_test_split(shot_train, test_size = 0.2, random_state = 42)

df_train = pd.DataFrame()
df_valid = pd.DataFrame()
df_test = pd.DataFrame()

for shot in shot_train:
    df_train = pd.concat([df_train, df[df.shot == shot]], axis = 0)

for shot in shot_valid:
    df_valid = pd.concat([df_valid, df[df.shot == shot]], axis = 0)

for shot in shot_test:
    df_test = pd.concat([df_test, df[df.shot == shot]], axis = 0)

scaler = RobustScaler()
df_train[ts_cols] = scaler.fit_transform(df_train[ts_cols].values)
df_valid[ts_cols] = scaler.transform(df_valid[ts_cols].values)
df_test[ts_cols] = scaler.transform(df_test[ts_cols].values)

# disruption info
kstar_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_extend.csv', encoding = "euc-kr")

# shot list
shot_list = np.unique(df.shot.values).tolist()

ts_train = df_train
ts_valid = df_valid
ts_test = df_test
seq_len = 21
dist = 1
dt = 1 / 210 * 4
cols = ts_cols
col_len = len(cols)
save_best_dir = "./weights/ts_transformer_clip_21_dist_1_best.pt"
save_last_dir = "./weights/ts_transformer_clip_21_dist_1_best.pt"
save_txt = "./results/test_ts_transformer_clip_21_dist_1.txt"
save_conf = "./results/test_ts_transformer_clip_21_dist_1_confusion_matrix.png"
save_latent_2d = "./results/ts_transformer_latent_2d.png"

train_data = DatasetFor0D(ts_train, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = 1.0 / 210 * 4)
valid_data = DatasetFor0D(ts_valid, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = 1.0 / 210 * 4)
test_data = DatasetFor0D(ts_test, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = 1.0 / 210 * 4)

from torch.utils.data import DataLoader
from src.utils.sampler import ImbalancedDatasetSampler

batch_size = 128
lr = 1e-3
gamma = 0.95
num_epoch = 64
verbose = 8

sampler = ImbalancedDatasetSampler(train_data)
train_loader = DataLoader(train_data, batch_size = batch_size, num_workers =8, sampler = sampler)
valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = 8, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 8, shuffle = True)

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

    model = TStransformer(
        n_features=col_len,
        feature_dims = 16,
        max_len = seq_len, 
        n_layers = 4,
        n_heads = 4, 
        dim_feedforward = 1024,
        dropout = 0.5, 
        cls_dims = 128, 
        n_classes  = 2
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=gamma)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma=gamma)

    train_data.get_num_per_cls()
    cls_num_list = train_data.get_cls_num_list()
    
    per_cls_weights = 1.0 / np.array(cls_num_list)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)

    focal_gamma = 2.0
    loss_fn = FocalLoss(weight = per_cls_weights, gamma = focal_gamma)

    '''
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

    # plot probability curve
    generate_prob_curve_from_0D(
        model, 
        batch_size = 4, 
        device = device, 
        save_dir = "./results/disruption_probs_curve.png",
        ts_data = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
        ts_cols = ts_cols,
        shot_list_dir = './dataset/KSTAR_Disruption_Shot_List_extend.csv',
        shot = 21310,
        seq_len = seq_len,
        dist = dist,
        dt = dt
    )
    
    # plot 2d latent space
    visualize_2D_latent_space(
        model,
        train_loader,
        device,
        save_latent_2d
    )