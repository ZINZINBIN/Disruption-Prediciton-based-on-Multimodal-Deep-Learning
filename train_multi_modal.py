import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

kstar_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_extend.csv', encoding = "euc-kr")
ts_data = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_for_multi.csv")
mult_info = pd.read_csv("./dataset/KSTAR_Disruption_multi_data.csv")

from torch.utils.data import Dataset
from typing import Optional, Literal, List, Union
from tqdm.auto import tqdm
from src.CustomDataset import DEFAULT_TS_COLS
import os, cv2

class MultiModalDataset(Dataset):
    def __init__(
        self, 
        task : Literal["train", "valid", "test"] = "train", 
        ts_data : Optional[pd.DataFrame] = None,
        ts_cols : Optional[List] = None,
        mult_info : Optional[pd.DataFrame] = None,
        dt : Optional[float] = 1.0 / 210 * 4,
        distance : Optional[int] = 0,
        n_fps : Optional[int] = 4,
        resize_height : Optional[int] = 256,
        resize_width : Optional[int] = 256,
        crop_size : Optional[int] = 128,
        seq_len : int = 21,
        n_classes : int = 2,
        ):
        self.task = task # task : train / valid / test 
        
        # resize each frame from video
        self.resize_height = resize_height
        self.resize_width = resize_width
        
        # crop
        self.crop_size = crop_size
        
        # video sequence length
        # warning : 0D data and video data should have equal sequence length
        self.seq_len = seq_len
        
        # use for 0D data prediction
        self.distance = distance # prediction time
        self.dt = dt # time difference of 0D data
        self.n_fps = n_fps

        # video_file_path : video file path : {database}/{shot_num}_{frame_start}_{frame_end}.avi
        # indices : index for tabular data, shot == shot_num, index <- df[df.frame_idx == frame_start].index
        self.n_classes = n_classes

        self.ts_data = ts_data
        self.mult_info = mult_info
        self.ts_cols = ts_cols
        
        # select columns for 0D data prediction
        if ts_cols is None:
            self.ts_cols = DEFAULT_TS_COLS
            
        self.video_file_path = mult_info[mult_info.task == task]["path"].values.tolist()
        self.labels = [0 if label is True else 1 for label in mult_info[mult_info.task == task].is_disrupt]
        self.indices = mult_info[mult_info.task == task]["t_start_index"].astype(int).values.tolist()

    def load_frames(self, file_dir : str):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = self.seq_len
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        
        for i, frame_name in enumerate(frames[::-1][::self.n_fps][::-1]):
            frame = np.array(cv2.imread(frame_name)).astype(np.float32)
            buffer[i] = frame
    
        return buffer
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx : int):
        x_video = self.get_video_data(idx)
        x_tabular = self.get_tabular_data(idx)
        label = torch.from_numpy(np.array(self.labels[idx]))
        return x_video, x_tabular, label

    def get_video_data(self, index : int):
        buffer = self.load_frames(self.video_file_path[index])
        if buffer.shape[0] < self.seq_len:
            buffer = self.refill_temporal_slide(buffer)
        buffer = self.crop(buffer, self.seq_len, self.crop_size)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer)
    
    def get_tabular_data(self, index : int):
        ts_idx = self.indices[index]
        data = self.ts_data[self.ts_cols].loc[ts_idx:ts_idx+self.seq_len-1].values
        return torch.from_numpy(data).float()

    def refill_temporal_slide(self, buffer:np.ndarray):
        for _ in range(self.seq_len - buffer.shape[0]):
            frame_new = buffer[-1].reshape(1, self.resize_height, self.resize_width, 3)
            buffer = np.concatenate((buffer, frame_new))
        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def to_tensor(self, buffer:Union[np.ndarray, torch.Tensor]):
        return buffer.transpose((3, 0, 1, 2))

    def crop(self, buffer : Union[np.ndarray, torch.Tensor], clip_len : int, crop_size : int, is_random : bool = False):
        if buffer.shape[0] < clip_len :
            time_index = np.random.randint(abs(buffer.shape[0] - clip_len))
        elif buffer.shape[0] == clip_len :
            time_index = 0
        else :
            time_index = np.random.randint(buffer.shape[0] - clip_len)

        if not is_random:
            original_height = self.resize_height
            original_width = self.resize_width
            mid_x, mid_y = original_height // 2, original_width // 2
            offset_x, offset_y = crop_size // 2, crop_size // 2
            buffer = buffer[time_index : time_index + clip_len, mid_x - offset_x:mid_x+offset_x, mid_y - offset_y: mid_y+ offset_y, :]
        else:
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)

            buffer = buffer[time_index:time_index + clip_len,
                    height_index:height_index + crop_size,
                    width_index:width_index + crop_size, :]
        return buffer

    # function for imbalanced dataset
    # used for LDAM loss and re-weighting
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
    

train_data = MultiModalDataset('train', ts_data, DEFAULT_TS_COLS, mult_info, dt = 1 / 210 * 4, distance = 4, seq_len = 21)
valid_data = MultiModalDataset('valid', ts_data, DEFAULT_TS_COLS, mult_info, dt = 1 / 210 * 4, distance = 4, seq_len = 21)
test_data = MultiModalDataset('test', ts_data, DEFAULT_TS_COLS, mult_info, dt = 1 / 210 * 4, distance = 4, seq_len = 21)

from torch.utils.data import DataLoader
from src.utils.sampler import ImbalancedDatasetSampler

batch_size = 32
sampler = ImbalancedDatasetSampler(train_data)
train_loader = DataLoader(train_data, batch_size = batch_size, num_workers = 8, sampler = sampler)
valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = 8, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 8, shuffle = True)

sample_video, sample_0D, sample_target = next(iter(train_loader))
print("sample_video : ", sample_video.size())
print("sample_0D : ", sample_0D.size())
print("sample_target : ", sample_target.size())


from typing import Optional, List, Literal, Union
from src.loss import LDAMLoss, FocalLoss
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

def train_per_epoch(
    train_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    max_norm_grad : Optional[float] = None
    ):

    model.train()
    model.to(device)

    train_loss = 0
    train_acc = 0

    total_pred = np.array([])
    total_label = np.array([])
    total_size = 0

    for batch_idx, (x_video, x_0D, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x_video = x_video.to(device)
        x_0D = x_0D.to(device)
        target = target.to(device)
        
        output = model(x_video, x_0D)
        loss = loss_fn(output, target)

        loss.backward()
        
        # use gradient clipping
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)

        optimizer.step()

        train_loss += loss.item()

        pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
        train_acc += pred.eq(target.view_as(pred)).sum().item()
        total_size += x_video.size(0) 
        
        total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
        total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))
        
    if scheduler:
        scheduler.step()

    train_loss /= total_size
    train_acc /= total_size

    train_f1 = f1_score(total_label, total_pred, average = "macro")

    return train_loss, train_acc, train_f1

def valid_per_epoch(
    valid_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    model.to(device)
    valid_loss = 0
    valid_acc = 0

    total_pred = np.array([])
    total_label = np.array([])
    total_size = 0

    for batch_idx, (x_video, x_0D, target) in enumerate(valid_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            x_video = x_video.to(device)
            x_0D = x_0D.to(device)
            target = target.to(device)
        
            output = model(x_video, x_0D)

            loss = loss_fn(output, target)
    
            valid_loss += loss.item()
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            valid_acc += pred.eq(target.view_as(pred)).sum().item()
            total_size += x_video.size(0) 

            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    valid_loss /= total_size
    valid_acc /= total_size

    valid_f1 = f1_score(total_label, total_pred, average = "macro")

    return valid_loss, valid_acc, valid_f1

def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : Union[torch.nn.CrossEntropyLoss, LDAMLoss, FocalLoss],
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_dir : str = "./weights/best.pt",
    save_last_dir : str = "./weights/last.pt",
    max_norm_grad : Optional[float] = None,
    criteria : Literal["f1_score", "acc", "loss"] = "f1_score",
    ):

    train_loss_list = []
    valid_loss_list = []
    
    train_acc_list = []
    valid_acc_list = []

    train_f1_list = []
    valid_f1_list = []

    best_acc = 0
    best_epoch = 0
    best_f1 = 0
    best_loss = torch.inf

    for epoch in tqdm(range(num_epoch), desc = "training process"):

        train_loss, train_acc, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad
        )

        valid_loss, valid_acc, valid_f1 = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device 
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train acc : {:.3f}, valid acc : {:.3f}, train f1 : {:.3f}, valid f1 : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_acc, valid_acc, train_f1, valid_f1
                ))

        # save the best parameters
        
        if criteria == "acc" and best_acc < valid_acc:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best_dir)
        elif criteria == "f1_score" and best_f1 < valid_f1:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best_dir)
        elif criteria == "loss" and best_loss > valid_loss:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best_dir)

        # save the last parameters
        torch.save(model.state_dict(), save_last_dir)

    # print("\n============ Report ==============\n")
    print("training process finished, best loss : {:.3f} and best acc : {:.3f}, best f1 : {:.3f}, best epoch : {}".format(
        best_loss, best_acc, best_f1, best_epoch
    ))

    return  train_loss_list, train_acc_list, train_f1_list,  valid_loss_list,  valid_acc_list, valid_f1_list

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true)**2)

def evaluate(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : Optional[torch.optim.Optimizer],
    loss_fn : Optional[torch.nn.Module]= None,
    device : Optional[str] = "cpu",
    save_conf : Optional[str] = "./results/confusion_matrix.png",
    save_txt : Optional[str] = None,
    threshold : float = 0.5,
    ):

    test_loss = 0
    test_acc = 0
    test_f1 = 0
    total_pred = np.array([])
    total_label = np.array([])

    if device is None:
        device = torch.device("cuda:0")

    model.to(device)
    model.eval()

    total_size = 0

    for idx, (x_video, x_0D, target) in enumerate(test_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            x_video = x_video.to(device)
            x_0D = x_0D.to(device)
            target = target.to(device)
            output = model(x_video, x_0D)
            loss = loss_fn(output, target)
    
            test_loss += loss.item()
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            pred = (pred > torch.FloatTensor([threshold]).to(device))
            test_acc += pred.eq(target.view_as(pred)).sum().item()

            total_size += x_video.size(0)
            
            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    test_loss /= (idx + 1)
    test_acc /= total_size
    test_f1 = f1_score(total_label, total_pred, average = "macro")
    test_auc = roc_auc_score(total_label, total_pred, average='macro')
    
    conf_mat = confusion_matrix(total_label, total_pred)

    if save_conf is None:
        save_conf = "./results/confusion_matrix.png"

    plt.figure()
    s = sns.heatmap(
        conf_mat, # conf_mat / np.sum(conf_mat),
        annot = True,
        fmt ='04d' ,# fmt = '.2f',
        cmap = 'Blues',
        xticklabels=["disruption","normal"],
        yticklabels=["disruption","normal"],
    )

    s.set_xlabel("Prediction")
    s.set_ylabel("Actual")

    plt.savefig(save_conf)

    print("############### Classification Report ####################")
    print(classification_report(total_label, total_pred, labels = [0,1]))
    print("\n# test acc : {:.2f}, test f1 : {:.2f}, test AUC : {:.2f}, test loss : {:.3f}".format(test_acc, test_f1, test_auc, test_loss))

    if save_txt:
        with open(save_txt, 'w') as f:
            f.write(classification_report(total_label, total_pred, labels = [0,1]))
            summary = "\n# test score : {:.2f}, test loss : {:.3f}, test f1 : {:.3f}, test_auc : {:.3f}".format(test_acc, test_loss, test_f1, test_auc)
            f.write(summary)

    return test_loss, test_acc, test_f1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 1e-3

from src.loss import FocalLoss
from src.models.mult_modal import MultiModalModel

train_data.get_num_per_cls()
cls_num_list = train_data.get_cls_num_list()
per_cls_weights = 1.0 / np.array(cls_num_list)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
loss_fn = FocalLoss(per_cls_weights, gamma = 2)

args_video = {
    "image_size" : 128, 
    "patch_size" : 32, 
    "n_frames" : 21, 
    "dim": 64, 
    "depth" : 4, 
    "n_heads" : 8, 
    "pool" : 'cls', 
    "in_channels" : 3, 
    "d_head" : 64, 
    "dropout" : 0.25,
    "embedd_dropout":  0.25, 
    "scale_dim" : 4
}

args_0D = {
    "seq_len" : 21, 
    "col_dim" : 9, 
    "conv_dim" : 32, 
    "conv_kernel" : 3,
    "conv_stride" : 1, 
    "conv_padding" : 1,
    "lstm_dim" : 64, 
}
    
model = MultiModalModel(
    2, args_video, args_0D
)

model.summary('cpu')
model.to(device)

num_epoch = 64
verbose = 4
save_best_dir = "./weights/multi_modal_clip_21_dist_8_best.pt"
save_last_dir = "./weights/multi_modal_clip_21_dist_8_last.pt"
save_conf = "./results/test_multi_modal_clip_21_dist_8_confusion_matrix.png"
save_txt = "./results/test_multi_modal_clip_21_dist_8.txt"
max_norm_grad = 1.0
criteria = "f1_score"
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train(
    train_loader,
    valid_loader,
    model,
    optimizer,
    None,
    loss_fn,
    device,
    num_epoch,
    verbose,
    save_best_dir,
    save_last_dir,
    max_norm_grad,
    criteria
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