import torch
import argparse
import numpy as np
import pandas as pd
from src.models.ConvLSTM import ConvLSTM
from src.utils.sampler import ImbalancedDatasetSampler
from src.utils.utility import plot_learning_curve, generate_prob_curve_from_0D
from torch.utils.data import DataLoader
from src.train import train
from src.evaluate import evaluate
from src.loss import LDAMLoss, FocalLoss

import numpy as np
import torch
import torch.nn as nn

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

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
scaler = RobustScaler()
df_train[ts_cols] = scaler.fit_transform(df_train[ts_cols].values)
df_valid[ts_cols] = scaler.transform(df_valid[ts_cols].values)
df_test[ts_cols] = scaler.transform(df_test[ts_cols].values)

# disruption info
kstar_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_extend.csv', encoding = "euc-kr")

# shot list
shot_list = np.unique(df.shot.values).tolist()

from typing import Optional, List
from tqdm.auto import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, ts_data : pd.DataFrame, disrupt_data : pd.DataFrame, seq_len : int, cols : List, dist:int, dt : float):
        self.ts_data = ts_data
        self.disrupt_data = disrupt_data
        self.seq_len = seq_len
        self.dt = dt
        self.cols = cols
        self.dist = dist # distance

        self.indices = []
        self.labels = []
        self.n_classes = 2
        self._generate_index()

    def _generate_index(self):
        shot_list = np.unique(self.ts_data.shot.values).tolist()
        df_disruption = self.disrupt_data

        for shot in tqdm(shot_list):
            tTQend = df_disruption[df_disruption.shot == shot].tTQend.values[0]
            tftsrt = df_disruption[df_disruption.shot == shot].tftsrt.values[0]
            tipminf = df_disruption[df_disruption.shot == shot].tipminf.values[0]

            t_disrupt = tipminf

            df_shot = self.ts_data[self.ts_data.shot == shot]
            indices = []
            labels = []

            idx = int(tftsrt * self.dt)
            idx_last = len(df_shot.index) - self.seq_len - self.dist

            while(idx < idx_last):
                row = df_shot.iloc[idx]
                t = row['time']

                if idx_last - idx - self.seq_len - self.dist < 0:
                    break

                if t >= tftsrt and t < t_disrupt - self.dt * (self.seq_len + self.dist):
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(0)
                    idx += self.seq_len

                elif t > t_disrupt - self.dt * (self.seq_len + self.dist) and t <= t_disrupt:
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(1)
                    idx += self.seq_len
                
                elif t < tftsrt:
                    idx += self.seq_len
                
                elif t > t_disrupt:
                    break

            self.indices.extend(indices)
            self.labels.extend(labels)

    def __getitem__(self, idx:int):
        indx = self.indices[idx]
        label = self.labels[idx]
        label = np.array(label)
        label = torch.from_numpy(label)
        data = self.ts_data[self.cols].loc[indx:indx+self.seq_len - 1].values
        data = torch.from_numpy(data).float()
        return data, label

    def __len__(self):
        return len(self.indices)

    def get_num_per_cls(self):
        classes = np.unique(self.labels)
        self.num_per_cls_dict = dict()

        for cls in classes:
            num = np.sum(np.where(self.labels == cls, 1, 0))
            self.num_per_cls_dict[cls] = num
         
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.n_classes):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

from sklearn.model_selection import train_test_split
ts_train = df_train
ts_valid = df_valid
ts_test = df_test
seq_len = 21
dist = 3
dt = 1 / 210 * 4
cols = ts_cols
col_len = len(cols)
save_best_dir = "./weights/ts_conv_lstm_clip_21_dist_3_best.pt"
save_last_dir = "./weights/ts_conv_lstm_clip_21_dist_3_best.pt"
save_txt = "./results/test_ts_transformer_clip_21_dist_3.txt"
save_conf = "./results/test_ts_transformer_clip_21_dist_3_confusion_matrix.png"

train_data = CustomDataset(ts_train, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = 1.0 / 210 * 4)
valid_data = CustomDataset(ts_valid, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = 1.0 / 210 * 4)
test_data = CustomDataset(ts_test, kstar_shot_list, seq_len = seq_len, cols = ts_cols, dist = dist, dt = 1.0 / 210 * 4)

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

    model = ConvLSTM(
    seq_len = seq_len,
    col_dim = col_len,
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