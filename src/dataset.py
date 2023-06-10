import os
import numpy as np
import pandas as pd
import torch
import random, glob2, cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, Dict, List, Union, Literal
from src.config import Config

# video data augmentation arguments(default)
DEFAULT_AUGMENTATION_ARGS = {
    "bright_val" : 10,
    "bright_p" : 0.25,
    "contrast_min" : 1,
    "contrast_max" : 1.15,
    "contrast_p" : 0.25,
    "blur_k" : 5,
    "blur_p" : 0.25,
    "flip_p" : 0.25,
    "vertical_ratio" : 0.1,
    "vertical_p" : 0.25,
    "horizontal_ratio" : 0.1,
    "horizontal_p" : 0.25
}

# 0D data default columns
config = Config()
DEFAULT_TS_COLS =  config.input_features

# Dataset for video model
class DatasetForVideo(Dataset):
    def __init__(
        self, 
        shot_dir_list : List,
        df_disrupt : pd.DataFrame,
        augmentation : Optional[bool] = True, 
        augmentation_args : Optional[Dict] = None,
        resize_height : Optional[int] = 256,
        resize_width : Optional[int] = 256,
        crop_size : Optional[int] = 224,
        seq_len : int = 21,
        dist : int = 3,
        ):

        self.shot_dir_list = shot_dir_list
        self.shot_list = [int(shot_dir.split("/")[-1]) for shot_dir in shot_dir_list]
        
        # to analyze which shot the network can not predict
        self.shot_num = []
        self.get_shot_num = False

        self.augmentation = augmentation # video sequence augmentation
        self.augmentation_args = augmentation_args # argument for augmentation
        
        # resize each frame from video
        self.resize_height = resize_height
        self.resize_width = resize_width
        
        # crop
        self.crop_size = crop_size
        
        # parameters for input and output data
        self.seq_len = seq_len
        self.dist = dist
        
        # data augmentation setup
        if augmentation_args is None:
            if self.augmentation is True:  
                self.augmentation_args = DEFAULT_AUGMENTATION_ARGS
            else:
                self.augmentation_args = None
        else:
            self.augmentation_args = augmentation_args

        self.video_file_path = []
        self.indices = []
        self.labels = []
        
        for shot_num, shot_dir in zip(self.shot_list, self.shot_dir_list):
            
            tipmin_frame = df_disrupt[df_disrupt.shot == shot_num]['frame_tipminf'].values.item()
            tftsrt_frame = df_disrupt[df_disrupt.shot == shot_num]['frame_startup'].values.item()
            dis_frame = tipmin_frame - dist
            
            indices = [i for i in reversed(range(dis_frame - seq_len, tftsrt_frame, -seq_len))]
            video_path = sorted(glob2.glob(os.path.join(shot_dir, "*")))
            
            for idx in indices:
                self.video_file_path.append(video_path[idx + 1 : idx + seq_len + 1])
                if idx == indices[-1]:
                    self.labels.append(0)
                else:
                    self.labels.append(1)
                    
            self.shot_num.extend([shot_num for _ in range(len(indices))])
        
        self.labels = np.array(self.labels, dtype=int)
        self.n_classes = 2
        
        self.n_disrupt = np.sum(self.labels==0)
        self.n_normal = np.sum(self.labels == 1)
        
    def load_frames(self, filepaths : List):
        buffer = np.empty((self.seq_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, filepath in enumerate(filepaths):
            frame = np.array(cv2.imread(filepath)).astype(np.float32) 
            buffer[i] = frame
            
        return buffer
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx : int):
        label = torch.from_numpy(np.array(self.labels[idx]))
        
        if self.get_shot_num:
            shot_num = self.shot_num[idx]
            return self.get_video_data(idx), label, shot_num
        else:
            return self.get_video_data(idx), label

    def get_video_data(self, index : int):
        buffer = self.load_frames(self.video_file_path[index])
        
        if buffer.shape[0] < self.seq_len:
            buffer = self.refill_temporal_slide(buffer)

        buffer = self.crop(buffer, self.seq_len, self.crop_size)

        if self.augmentation:
            buffer = self.brightness(buffer, val = self.augmentation_args["bright_val"], p = self.augmentation_args["bright_p"])
            buffer = self.contrast(buffer, self.augmentation_args["contrast_min"], self.augmentation_args["contrast_max"], p = self.augmentation_args["contrast_p"])
            buffer = self.blur(buffer, p = self.augmentation_args["blur_p"], kernel_size = self.augmentation_args["blur_k"])
            buffer = self.randomflip(buffer, p = self.augmentation_args["flip_p"])
            buffer = self.vertical_shift(buffer, ratio = self.augmentation_args["vertical_ratio"], p = self.augmentation_args["vertical_p"])
            buffer = self.horizontal_shift(buffer, ratio = self.augmentation_args["horizontal_ratio"], p = self.augmentation_args["horizontal_p"])

        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        buffer = torch.from_numpy(buffer)
 
        return buffer

    def refill_temporal_slide(self, buffer:np.ndarray):
        for _ in range(self.seq_len - buffer.shape[0]):
            frame_new = buffer[-1].reshape(1, self.resize_height, self.resize_width, 3)
            buffer = np.concatenate((buffer, frame_new))
        return buffer

    def randomflip(self, buffer, p :float = 0.5):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < p:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def horizontal_shift(self, buffer, ratio : float = 0.0, p : float = 0.5):
        if np.random.random() < p:
            ratio = random.uniform(-ratio, ratio)
            to_shift = int(self.crop_size * ratio)
            if ratio > 0:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[:,:-to_shift, :] = frame[:,:-to_shift, :]
                    buffer[i] = ref
            else:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[:,-to_shift:, :] = frame[:,-to_shift:, :]
                    buffer[i] = ref

        return buffer

    def vertical_shift(self, buffer, ratio : float = 0.0, p : float = 0.5):
        if np.random.random() < p:
            ratio = random.uniform(-ratio, ratio)
            to_shift = int(self.crop_size * ratio)
            if ratio > 0:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[:-to_shift, :, :] = frame[:-to_shift, :, :]
                    buffer[i] = ref
                    
            else:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[-to_shift:, :, :] = frame[-to_shift:, :, :]
                    buffer[i] = ref
        return buffer

    def blur(self, buffer, p : float = 0.5, kernel_size : int = 5):
        if np.random.random() < p:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def brightness(self, buffer, val : int = 30, p : float = 0.5):
        bright = int(random.uniform(-val, val))
        if np.random.random() < p:
            if bright > 0:
                for i, frame in enumerate(buffer):
                    frame = buffer[i] + bright
                    buffer[i] = np.clip(frame, 10, 255)
            else:
                for i, frame in enumerate(buffer):
                    frame = buffer[i] - bright
                    buffer[i] = cv2.flip(frame, flipCode=1)
            return buffer
        else:
            return buffer

    def contrast(self, buffer, min_val : float = 1.0, max_val : float = 1.5, p : float = 0.5):
        if np.random.random() < p:
            alpha = int(random.uniform(min_val, max_val))
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.convertScaleAbs(frame, alpha = alpha)
        return buffer

    def to_tensor(self, buffer:Union[np.ndarray, torch.Tensor]):
        return buffer.transpose((3, 0, 1, 2))

    def crop(self, buffer : Union[np.ndarray, torch.Tensor], clip_len : int, crop_size : int, is_random : bool = False):
        # randomly select time index for temporal jittering
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
            # Randomly select start indices in order to crop the video
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)

            buffer = buffer[
                time_index:time_index + clip_len,
                height_index:height_index + crop_size,
                width_index:width_index + crop_size,:]

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

# Dataset for 0D data : different method used for label matching
class DatasetFor0D(Dataset):
    def __init__(self, ts_data : pd.DataFrame, disrupt_data : pd.DataFrame, seq_len : int = 21, cols : List = DEFAULT_TS_COLS, dist:int = 3, dt : float = 1.0 / 210 * 4, scaler = None):
        self.ts_data = ts_data
        self.disrupt_data = disrupt_data
        self.seq_len = seq_len
        self.dt = dt
        self.cols = cols
        self.dist = dist # distance

        self.indices = []
        self.labels = []
        self.shot_num = [] # shot list with respond to indices
        self.get_shot_num = False
        
        self.scaler = scaler
        
        self.n_classes = 2
        
        shot_list = np.unique(self.ts_data.shot.values).tolist()
        self.shot_list = [shot_num for shot_num in shot_list if shot_num in disrupt_data.shot.values]
        
        self.preprocessing()
        self._generate_index()
        
    def preprocessing(self):
        shot_ignore = []
        
        # filter : remove invalid shot
        for shot in tqdm(self.shot_list, desc = 'remove invalid data : null / measurement error'):
            # 1st filter : remove null data
            df_shot = self.ts_data[self.ts_data.shot == shot]
            null_check = df_shot[self.cols].isna().sum()
            
            is_null = False
            
            for c in null_check:
                if c > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    is_null = True
                    break
            
            if is_null:
                continue
            
            # 2nd filter : measurement error
            for col in self.cols:
                if np.sum(df_shot[col] == 0) > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    break

                # constant value
                if df_shot[col].max() - df_shot[col].min() < 1e-3:
                    shot_ignore.append(shot)
                    break
        
        # update shot list with ignoring the null data
        shot_list_new = [shot_num for shot_num in self.shot_list if shot_num not in shot_ignore]
        self.shot_list = shot_list_new
        
        # 0D parameter : NAN -> forward fill
        for shot in tqdm(self.shot_list, desc = 'replace nan value'):
            df_shot = self.ts_data[self.ts_data.shot == shot].copy()
            self.ts_data.loc[self.ts_data.shot == shot, self.cols] = df_shot[self.cols].fillna(0)
            
        if self.scaler is not None:
            self.ts_data[self.cols] = self.scaler.transform(self.ts_data[self.cols])

    def _generate_index(self):
        df_disruption = self.disrupt_data
        
        for shot in tqdm(self.shot_list):
            tTQend = df_disruption[df_disruption.shot == shot].tTQend.values[0]
            tftsrt = df_disruption[df_disruption.shot == shot].tftsrt.values[0]
            tipminf = df_disruption[df_disruption.shot == shot].tipminf.values[0]
            t_disrupt = tipminf
            
            df_shot = self.ts_data[self.ts_data.shot == shot]
            
            indices = []
            labels = []

            idx = int(tftsrt / self.dt)
            idx_last = len(df_shot.index) - self.seq_len - self.dist
            
            while(idx < idx_last):
                row = df_shot.iloc[idx]
                t = row['time']

                if idx_last - idx < 0:
                    break

                if t >= tftsrt and t < t_disrupt - self.dt * (self.seq_len + self.dist):
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(1)
                    idx += self.seq_len // 3

                elif t >= t_disrupt - self.dt * (2 * self.seq_len + self.dist) and t < t_disrupt - self.dt * (self.seq_len + self.dist):
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(1)
                    idx += self.seq_len // 7
                
                elif t >= t_disrupt - self.dt * (self.seq_len + self.dist) and t <= t_disrupt - self.dt * self.seq_len + self.dt:
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(0)
                    idx += 1
                
                elif t < tftsrt:
                    idx += self.seq_len // 3
                
                elif t > t_disrupt:
                    break
                
                else:
                    idx += self.seq_len // 3
                
            self.shot_num.extend([shot for _ in range(len(indices))])
            self.indices.extend(indices)
            self.labels.extend(labels)
            
        self.n_disrupt = np.sum(np.array(self.labels)==0)
        self.n_normal = np.sum(np.array(self.labels)==1)
        
    def __getitem__(self, idx:int):
        indx = self.indices[idx]
        label = self.labels[idx]
        label = np.array(label)
        label = torch.from_numpy(label)
        data = self.ts_data[self.cols].loc[indx+1:indx+self.seq_len].values
        data = torch.from_numpy(data).float()
        
        if self.get_shot_num:
            shot_num = self.shot_num[idx]
            return data, label, shot_num
        
        else:
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
        data_video = self.get_video_data(idx)
        data_0D = self.get_tabular_data(idx)
        data_dict = {
            "video" : data_video,
            "0D" : data_0D
        }
        label = torch.from_numpy(np.array(self.labels[idx]))
        return data_dict, label

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

class MultiModalDataset2(Dataset):
    def __init__(
        self, 
        shot_dir_list : List,
        df_disrupt : pd.DataFrame, 
        ts_data : pd.DataFrame, 
        ts_cols : List,
        augmentation : Optional[bool] = True, 
        augmentation_args : Optional[Dict] = None,
        resize_height : Optional[int] = 256,
        resize_width : Optional[int] = 256,
        crop_size : Optional[int] = 224,
        seq_len : int = 21,
        dist : int = 3,
        dt : float = 1.0 / 210 * 4,
        scaler = None,
        tau : int = 1,
        ):
        
        # integer interval
        self.tau = tau
        
        self.n_classes = 2
        
        self.shot_dir_list = shot_dir_list
        self.shot_list = [int(shot_dir.split("/")[-1]) for shot_dir in shot_dir_list]
        
         # to analyze which shot the network can not predict
        self.shot_num = []
        self.get_shot_num = False

        self.augmentation = augmentation # video sequence augmentation
        self.augmentation_args = augmentation_args # argument for augmentation
        
        # resize each frame from video
        self.resize_height = resize_height
        self.resize_width = resize_width
        
        # crop
        self.crop_size = crop_size
        
        # video sequence length
        self.seq_len = seq_len
        self.dist = dist
        
        # use for 0D data prediction
        self.ts_data = ts_data
        self.ts_cols = ts_cols
        self.dt = dt # time difference of 0D data
        
        # scaler
        self.scaler = scaler
        
        # disruption info
        self.df_disrupt = df_disrupt
        
        # select columns for 0D data prediction
        if ts_cols is None:
            self.ts_cols = DEFAULT_TS_COLS
            
        # preprocessing for ts data
        # step 1. middle value nan -> interpolation
        self.ts_data[self.ts_cols] = self.ts_data[self.ts_cols].interpolate(method = "linear", limit_direction = 'forward')
        
        # step 2. nan for all value -> 0
        self.ts_data[self.ts_cols] = self.ts_data[self.ts_cols].fillna(method = 'ffill')
        
        # data scaling for ts data
        if self.scaler:
            self.ts_data[self.ts_cols] = scaler.transform(self.ts_data[self.ts_cols])
        
        # data augmentation setup
        if augmentation_args is None:
            if self.augmentation is True:  
                self.augmentation_args = DEFAULT_AUGMENTATION_ARGS
            else:
                self.augmentation_args = None
        else:
            self.augmentation_args = augmentation_args        
        
        # data - label generation
        # First : remove the shot which have too many nan values
        shot_list_ts = self.ts_data.shot.unique()
        self.shot_list = [shot for shot in self.shot_list if shot in shot_list_ts]
        
        shot_ignore = []
        for shot in tqdm(self.shot_list, desc = 'extract the null data / short time data'):
            df_shot = self.ts_data[self.ts_data.shot==shot]
            null_check = df_shot[self.ts_cols].isna().sum()
            
            t_max = max(df_shot.time.values)
            tipminf = df_disrupt[df_disrupt.shot == shot].tipminf.values[0]
            
            if t_max < tipminf - dist * self.dt:
                shot_ignore.append(shot)
                break
            
            for c in null_check:
                if c > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    break
        
        shot_list_tmp = []
        shot_dir_list_tmp = []
        for shot, shot_dir in zip(self.shot_list, self.shot_dir_list):
            if shot not in shot_ignore:
                shot_list_tmp.append(shot)
                shot_dir_list_tmp.append(shot_dir)
        
        self.shot_list = shot_list_tmp
        self.shot_dir_list = shot_dir_list_tmp
        
        # determine the disruptive phase
        self.video_file_path = []
        self.ts_data_indices = []
        
        self.indices = []
        self.labels = []
        
        for shot_num, shot_dir in zip(self.shot_list, self.shot_dir_list):
            
            # ts data per shot
            df_shot = self.ts_data[self.ts_data.shot == shot_num]
            
            # Thermal quench and current quench time / frame number
            tipminf = df_disrupt[df_disrupt.shot == shot_num].tipminf.values[0]
            tftsrt = df_disrupt[df_disrupt.shot == shot_num].tftsrt.values[0]
            
            tipmin_frame = df_disrupt[df_disrupt.shot == shot_num]['frame_tipminf'].values.item()
            tftsrt_frame = df_disrupt[df_disrupt.shot == shot_num]['frame_startup'].values.item()
            
            t_disrupt = tipminf - dist * self.dt
            dis_frame = tipmin_frame - dist 
            
            res = min(len(df_shot) - len(df_shot[df_shot.time > t_disrupt]), self.seq_len * self.tau)
            
            # indices for video and ts data per shot
            # video indices
            video_indices = [i for i in reversed(range(dis_frame - res, tftsrt_frame, -tau*seq_len//3))]
            
            # ts indices
            ts_idx_last = len(df_shot) - len(df_shot[df_shot.time > t_disrupt])
            ts_idx_start = int(tftsrt * self.dt)
            ts_indices = [i for i in reversed(range(ts_idx_last - res, ts_idx_start, -tau*seq_len//3))]
            
            # video path per shot
            video_path = sorted(glob2.glob(os.path.join(shot_dir, "*")))
            
            # ts indices
            ts_indices_tmp = []
            for idx in reversed(ts_indices):
                row = df_shot.iloc[idx]
                t = row['time']
                
                if t <= t_disrupt:
                    indx = df_shot.index.values[idx]
                    ts_indices_tmp.append(indx)
                    
            if len(ts_indices_tmp) > len(video_indices):
                ts_indices_tmp = ts_indices_tmp[0:len(video_indices)]
            elif len(ts_indices_tmp) < len(video_indices):
                video_indices = video_indices[0:len(ts_indices_tmp)]
            
            self.ts_data_indices.extend(reversed(ts_indices_tmp))
            
            # video indices
            for idx in video_indices:
                # with tau
                self.video_file_path.append(video_path[idx + tau*seq_len + 1:idx+1:-tau][::-1])
                
                if idx >= dis_frame - tau * self.seq_len - tau * self.seq_len // 6:
                    self.labels.append(0)
                else:
                    self.labels.append(1)
                    
            self.shot_num.extend([shot_num for _ in range(len(video_indices))])
            
        print("# check | video data : {}, 0D data : {}".format(len(self.video_file_path), len(self.ts_data_indices)))
        self.n_disrupt = np.sum(np.array(self.labels)==0)
        self.n_normal = np.sum(np.array(self.labels)==1)

    def load_frames(self, filepaths : List):
        buffer = np.empty((self.seq_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, filepath in enumerate(filepaths):
            frame = np.array(cv2.imread(filepath)).astype(np.float32) 
            buffer[i] = frame
        return buffer
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx : int):
        data_video = self.get_video_data(idx)
        data_0D = self.get_tabular_data(idx)
        data_dict = {
            "video" : data_video,
            "0D" : data_0D
        }
        label = torch.from_numpy(np.array(self.labels[idx]))
        
        if self.get_shot_num:
            shot_num = self.shot_num[idx]
            return data_dict, label, shot_num
        else:
            return data_dict, label

    def get_video_data(self, index : int):
        buffer = self.load_frames(self.video_file_path[index])
        
        if buffer.shape[0] < self.seq_len:
            buffer = self.refill_temporal_slide(buffer)

        buffer = self.crop(buffer, self.seq_len, self.crop_size)

        if self.augmentation:
            buffer = self.brightness(buffer, val = self.augmentation_args["bright_val"], p = self.augmentation_args["bright_p"])
            buffer = self.contrast(buffer, self.augmentation_args["contrast_min"], self.augmentation_args["contrast_max"], p = self.augmentation_args["contrast_p"])
            buffer = self.blur(buffer, p = self.augmentation_args["blur_p"], kernel_size = self.augmentation_args["blur_k"])
            buffer = self.randomflip(buffer, p = self.augmentation_args["flip_p"])
            buffer = self.vertical_shift(buffer, ratio = self.augmentation_args["vertical_ratio"], p = self.augmentation_args["vertical_p"])
            buffer = self.horizontal_shift(buffer, ratio = self.augmentation_args["horizontal_ratio"], p = self.augmentation_args["horizontal_p"])

        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        buffer = torch.from_numpy(buffer)
 
        return buffer
    
    def get_tabular_data(self, index : int):
        ts_idx = self.ts_data_indices[index]
        data = self.ts_data[self.ts_cols].loc[ts_idx+1:ts_idx+self.seq_len*self.tau].values[::self.tau,:]
        return torch.from_numpy(data).float()

    def refill_temporal_slide(self, buffer:np.ndarray):
        for _ in range(self.seq_len - buffer.shape[0]):
            frame_new = buffer[-1].reshape(1, self.resize_height, self.resize_width, 3)
            buffer = np.concatenate((buffer, frame_new))
        return buffer

    def to_tensor(self, buffer:Union[np.ndarray, torch.Tensor]):
        return buffer.transpose((3, 0, 1, 2))
    
    def randomflip(self, buffer, p :float = 0.5):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < p:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def horizontal_shift(self, buffer, ratio : float = 0.0, p : float = 0.5):
        if np.random.random() < p:
            ratio = random.uniform(-ratio, ratio)
            to_shift = int(self.crop_size * ratio)
            if ratio > 0:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[:,:-to_shift, :] = frame[:,:-to_shift, :]
                    buffer[i] = ref
            else:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[:,-to_shift:, :] = frame[:,-to_shift:, :]
                    buffer[i] = ref

        return buffer

    def vertical_shift(self, buffer, ratio : float = 0.0, p : float = 0.5):
        if np.random.random() < p:
            ratio = random.uniform(-ratio, ratio)
            to_shift = int(self.crop_size * ratio)
            if ratio > 0:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[:-to_shift, :, :] = frame[:-to_shift, :, :]
                    buffer[i] = ref
                    
            else:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[-to_shift:, :, :] = frame[-to_shift:, :, :]
                    buffer[i] = ref
        return buffer

    def blur(self, buffer, p : float = 0.5, kernel_size : int = 5):
        if np.random.random() < p:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def brightness(self, buffer, val : int = 30, p : float = 0.5):
        bright = int(random.uniform(-val, val))
        if np.random.random() < p:
            if bright > 0:
                for i, frame in enumerate(buffer):
                    frame = buffer[i] + bright
                    buffer[i] = np.clip(frame, 10, 255)
            else:
                for i, frame in enumerate(buffer):
                    frame = buffer[i] - bright
                    buffer[i] = cv2.flip(frame, flipCode=1)
            return buffer
        else:
            return buffer

    def contrast(self, buffer, min_val : float = 1.0, max_val : float = 1.5, p : float = 0.5):
        if np.random.random() < p:
            alpha = int(random.uniform(min_val, max_val))
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.convertScaleAbs(frame, alpha = alpha)
        return buffer

    def to_tensor(self, buffer:Union[np.ndarray, torch.Tensor]):
        return buffer.transpose((3, 0, 1, 2))

    def crop(self, buffer : Union[np.ndarray, torch.Tensor], clip_len : int, crop_size : int, is_random : bool = False):
        # randomly select time index for temporal jittering
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
            # Randomly select start indices in order to crop the video
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)

            buffer = buffer[
                time_index:time_index + clip_len,
                height_index:height_index + crop_size,
                width_index:width_index + crop_size,:]

        return buffer

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

if __name__ == "__main__":

    test_data = DatasetForVideo(
        root_dir = "./dataset/dur21_dis0",
        task = 'train',
        augmentation=False,
        augmentation_args=None,
        resize_height = 256,
        resize_width=256,
        crop_size = 224,
        seq_len = 21,
    )

    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    sample_data, sample_label = next(iter(test_loader))

    print("test for video loader")
    print('sample_data : ', sample_data.size())
    print('sample_target : ', sample_label.size())
    
    del test_loader, test_data, sample_data, sample_label
    
    test_data = DatasetFor0D(
        pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index(), 
        pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_extend.csv', encoding = "euc-kr"), 
        seq_len = 21, 
        cols = DEFAULT_TS_COLS, 
        dist = 3, 
        dt = 1.0 / 210 * 4
    )

    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    sample_data, sample_label = next(iter(test_loader))

    print("test for 0D data loader")
    print('sample_data : ', sample_data.size())
    print('sample_target : ', sample_label.size())