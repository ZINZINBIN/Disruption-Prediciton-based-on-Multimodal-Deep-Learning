import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import cv2, os, glob2, random
import pandas as pd
import numpy as np
from typing import Optional, List, Literal, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator
from src.config import Config
import time
import gc

config = Config()
STATE_FIXED = config.STATE_FIXED

# For reproduction
def seed_everything(seed : int = 42, deterministic : bool = False):
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        
# function for deterministic train_test_split function
def deterministic_split(shot_list : List, test_size : float = 0.2):
    n_length = len(shot_list)
    n_test = int(test_size * n_length)
    divided = n_length // n_test

    total_indices = [idx for idx in range(n_length)]
    test_indices = [idx for idx in range(n_length) if idx % divided == 0]
    
    train_list = []
    test_list = []
    
    for idx in total_indices:
        if idx in test_indices:
            test_list.append(shot_list[idx])
        else:
            train_list.append(shot_list[idx])
    
    return train_list, test_list

# train-test split for video data model
def preparing_video_dataset(root_dir : str, random_state : int = STATE_FIXED, test_shot : Optional[int] = 21310):
    shot_list = glob2.glob(os.path.join(root_dir, "*"))
    
    if test_shot is not None:
        shot_list = [shot_dir for shot_dir in shot_list if str(test_shot) not in shot_dir]
    
    # stochastic train_test_split
    # shot_train, shot_test = train_test_split(shot_list, test_size = 0.2, random_state = random_state)
    # shot_train, shot_valid = train_test_split(shot_train, test_size = 0.2, random_state = random_state) 
    
    # deterministic train_test_split
    shot_train, shot_test = deterministic_split(shot_list, test_size = 0.2)
    shot_train, shot_valid = deterministic_split(shot_train, test_size = 0.2)
    
    return shot_train, shot_valid, shot_test

# train-test split for 0D data models
def preparing_0D_dataset(filepath : str = "./dataset/KSTAR_Disruption_ts_data_extend.csv", random_state : int = STATE_FIXED, ts_cols : Optional[List] = None, scaler : Literal['Robust', 'Standard', 'MinMax'] = 'Robust', test_shot : Optional[int] = 21310):
    
    # preparing 0D data for use
    df = pd.read_csv(filepath).reset_index()

    # nan interpolation
    df.interpolate(method = 'linear', limit_direction = 'forward')

    # float type
    if ts_cols is None:
        for col in df.columns:
            df[col] = df[col].astype(np.float32)
    else:
        for col in ts_cols:
            df[col] = df[col].astype(np.float32)

    # train / valid / test data split
    shot_list = np.unique(df.shot.values)
    
    if test_shot is not None:
        shot_list = np.array([shot for shot in shot_list if int(shot) != test_shot])

    # stochastic train_test_split
    # shot_train, shot_test = train_test_split(shot_list, test_size = 0.2, random_state = random_state)
    # shot_train, shot_valid = train_test_split(shot_train, test_size = 0.2, random_state = random_state)
    
    # deterministic train_test_split
    shot_train, shot_test = deterministic_split(shot_list, test_size = 0.2)
    shot_train, shot_valid = deterministic_split(shot_train, test_size = 0.2)
    
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for shot in shot_train:
        df_train = pd.concat([df_train, df[df.shot == shot]], axis = 0)

    for shot in shot_valid:
        df_valid = pd.concat([df_valid, df[df.shot == shot]], axis = 0)

    for shot in shot_test:
        df_test = pd.concat([df_test, df[df.shot == shot]], axis = 0)
        
    if scaler == 'Robust':
        scaler = RobustScaler()
    elif scaler == 'Standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    scaler.fit(df_train[ts_cols].values)

    return df_train, df_valid, df_test, scaler
    
def preparing_multi_data(root_dir : str, ts_filepath : str = "./dataset/KSTAR_Disruption_ts_data_5ms.csv", ts_cols : Optional[List] = None, scaler : Literal['Robust', 'Standard', 'MinMax'] = 'Robust', test_shot : Optional[int] = 21310):
    
    shot_list = glob2.glob(os.path.join(root_dir, "*"))
    
    if test_shot is not None:
        shot_list = [shot_dir for shot_dir in shot_list if str(test_shot) not in shot_dir]
    
    shot_train, shot_test = train_test_split(shot_list, test_size = 0.2, random_state = 42)
    shot_train, shot_valid = train_test_split(shot_train, test_size = 0.2, random_state = 42) 
    
    # preparing 0D data for use
    df = pd.read_csv(ts_filepath).reset_index()

    # nan interpolation
    df.interpolate(method = 'linear', limit_direction = 'forward')

    # float type
    if ts_cols is None:
        for col in df.columns:
            df[col] = df[col].astype(np.float32)
    else:
        for col in ts_cols:
            df[col] = df[col].astype(np.float32)

    # train / valid / test data split
    ts_shot_train_list = [int(shot_dir.split("/")[-1]) for shot_dir in shot_train]
    ts_shot_valid_list = [int(shot_dir.split("/")[-1]) for shot_dir in shot_valid]
    ts_shot_test_list = [int(shot_dir.split("/")[-1]) for shot_dir in shot_test]

    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for shot in ts_shot_train_list:
        df_train = pd.concat([df_train, df[df.shot == shot]], axis = 0)

    for shot in ts_shot_valid_list:
        df_valid = pd.concat([df_valid, df[df.shot == shot]], axis = 0)

    for shot in ts_shot_test_list:
        df_test = pd.concat([df_test, df[df.shot == shot]], axis = 0)
        
    if scaler == 'Robust':
        scaler = RobustScaler()
    elif scaler == 'Standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    scaler.fit(df_train[ts_cols].values)

    return (shot_train, df_train), (shot_valid, df_valid), (shot_test, df_test), scaler
    
    
def preprocessing_video(file_path : str, width : int = 256, height: int = 256, overwrite : bool = True, save_path : Optional[str] = None):
    '''
    preprocessing_video : load video data by cv2 to save as resized image file(.jpg)
    - file_path : directory for video file
    - width, height : resized image file width, height
    - overwrite : if true, save file as image(.jpg) from save_dir
    - save_dir : if overwrite, save file to save_dir
    '''
    video_filename = file_path.split('.')[0]

    if os.path.isfile(file_path):
        capture = cv2.VideoCapture(file_path)
    else:
        capture = None
        raise "file_path is not valid, video data can not be found"

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0
    retaining = True

    while(count < frame_count and retaining):
        retaining, frame = capture.read()

        if frame is None:
            continue

        if frame_height != height or frame_width != width:
            frame = cv2.resize(frame, (width, height))

        if overwrite and save_path is not None:
            cv2.imwrite(filename = os.path.join(save_path, video_filename, '0000{}.jpg'.format(str(count))), img = frame)
        count += 1
    
    capture.release()

# plot the frame as image 
def show_frame(frame):
    if type(frame) == torch.Tensor:
        frame_img = frame.numpy()
    else:
        frame_img = frame
    
    if len(frame_img.shape) == 4:
        frame_img = np.squeeze(frame_img, axis  = 0)

    if frame_img.shape[2] != 3 or frame_img.shape[0] == 3:
        frame_img = np.transpose(frame_img, (1,2,0))
    
    plt.imshow(frame_img)
    plt.show()


def load_frames(file_path : str, height : int = 256, width : int = 256, channel : int = 3, to_tensor : bool = True):
    '''load video data from file_path, (optional : convert to tensor)
    - file_path : file path for video data
    - height : resized img height
    - width : resized img width
    - channel : RGB(channel = 3)
    - to_tensor : if true, return tensor type
    '''
    frame_list = sorted([os.path.join(file_path, img_path) for img_path in os.listdir(file_path)])
    frame_count = len(frame_list)

    buffer = np.empty((frame_count, height, width, channel), np.dtype('float32'))

    for idx, frame_path in enumerate(frame_list):
        frame = np.array(cv2.imread(frame_path)).astype(np.float32)
        buffer[idx] = frame
    
    if to_tensor:
        buffer = torch.from_numpy(buffer).dtype(torch.float32)
    
    return buffer


def load_frames_with_interval(file_path : str, height : int = 256, width : int = 256, channel : int = 3, interval : int = 6, to_tensor : bool = True):
    '''load video data from file_path with interval(optional : convert to tensor)
    - file_path : file path for video data
    - height : resized img height
    - width : resized img width
    - channel : RGB(channel = 3)
    - to_tensor : if true, return tensor type
    - interval : interval between two adjacent frames
    '''
    frame_list = sorted([os.path.join(file_path, img_path) for img_path in os.listdir(file_path)])
    frame_count = int(len(frame_list) / interval)

    buffer = np.empty((frame_count, height, width, channel), np.dtype('float32'))

    count = 0

    while count <= frame_count:
        frame_path = frame_list[count * interval]
        frame = np.array(cv2.imread(frame_path)).astype(np.float32)
        buffer[count] = frame
        count += 1
    
    if to_tensor:
        buffer = torch.from_numpy(buffer).dtype(torch.float32)
    
    return buffer


def crop(buffer, original_height, original_width, crop_size):
    mid_x, mid_y = original_height // 2, original_width // 2
    offset_x, offset_y = crop_size // 2, crop_size // 2
    buffer = buffer[:, mid_x - offset_x:mid_x+offset_x, mid_y - offset_y: mid_y+ offset_y, :]

    return buffer

def normalize(buffer:np.ndarray):
    for i, frame in enumerate(buffer):
        frame -= np.array([[[90.0, 98.0, 102.0]]])
        buffer[i] = frame
    return buffer

def time_split(buffer:np.ndarray, clip_len : int, use_continuous_frame : bool = True):
    frame_count = buffer.shape[0]
    h = buffer.shape[1]
    w = buffer.shape[2]
    c = buffer.shape[3]

    if use_continuous_frame:
        batch_size = frame_count - clip_len + 1
        dataset = np.empty((batch_size, clip_len, h, w, c), dtype = np.float32)

        for idx in range(0, batch_size):
            t_start = idx
            t_end = idx + clip_len
            dataset[idx, :, :, :, :] = buffer[t_start : t_end, :, :, :]
    else:
        batch_size = frame_count // clip_len
        batch_rest = frame_count % clip_len
        if batch_rest != 0:
            dataset = np.zeros((batch_size+1, clip_len, h, w, c), dtype = np.float32)
        else:
            dataset = np.zeros((batch_size, clip_len, h, w, c), dtype = np.float32)

        for idx in range(0, dataset.shape[0]):
            t_start = idx * clip_len
            t_end = t_start + clip_len

            if idx == dataset.shape[0] - 1 and batch_rest != 0:
                t_end = t_start + batch_rest
                dataset[idx, 0:batch_rest, :, :, :] = buffer[t_start : t_end, :, :, :]
            else:
                dataset[idx, :, :, :, :] = buffer[t_start : t_end, :, :, :]
    
    return dataset.transpose((0, 4, 1, 2, 3))

# generate video to input data
def video2tensor(
    dir : str, 
    channels : int = 3, 
    clip_len : int = 42, 
    crop_size : int = 112,
    resize_width : int = 171,
    resize_height : int = 128,
    use_continuous_frame : bool = True
    ):

    capture = cv2.VideoCapture(dir)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0
    retaining = True

    # build buffer with (frame_count, resize_width, resize_height, c)
    buffer = np.empty((frame_count, resize_height, resize_width, channels), np.dtype('float32'))

    while (count < frame_count and retaining):
        retaining, frame = capture.read()

        if frame is None:
            frame = np.zeros((resize_width, resize_height, channels))

        if (frame_height != resize_height) or (frame_width != resize_width):
            frame = cv2.resize(frame, (resize_width, resize_height))

        buffer[count] = frame
        count += 1

    capture.release()

    buffer = crop(buffer, resize_height, resize_width, crop_size)
    buffer = normalize(buffer)
    dataset = time_split(buffer, clip_len, use_continuous_frame)
    dataset = torch.from_numpy(dataset)
    return dataset

# VideoDataset for generating probability curve
class VideoDataset(Dataset):
    def __init__(
        self, 
        root_dir : Optional[str],
        resize_height : Optional[int] = 256,
        resize_width : Optional[int] = 256,
        crop_size : Optional[int] = 224,
        seq_len : int = 21,
        dist : int = 1,
        frame_srt : int = 0,
        frame_end : int = -1,
        ):
        
        self.root_dir = root_dir # video root directory
        # resize each frame from video
        self.resize_height = resize_height
        self.resize_width = resize_width
        
        # crop
        self.crop_size = crop_size
        
        # video sequence length
        self.seq_len = seq_len
   
        # indices : index for tabular data, shot == shot_num, index <- df[df.frame_idx == frame_start].index
        self.paths = glob2.glob(os.path.join(root_dir, "*"))
        self.original_path = sorted(glob2.glob(os.path.join(root_dir, "*")))
        
        self.paths = sorted(self.paths)[frame_srt:frame_end + 210]
        # self.paths = sorted(self.paths)
        
        self.path_indices = [idx for idx in range(0,len(self.paths)-seq_len-dist)]

    def load_frames(self, idx : int):
        idx_srt = self.path_indices[idx]
        idx_end = idx_srt + self.seq_len

        frames = sorted(self.paths[idx_srt : idx_end])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float32)
            buffer[i] = frame
        return buffer
    
    def __len__(self):
        return len(self.path_indices)

    def __getitem__(self, idx : int):
        return self.get_video_data(idx)

    def get_video_data(self, index : int):
        buffer = self.load_frames(index)
        
        if buffer.shape[0] < self.seq_len:
            buffer = self.refill_temporal_slide(buffer)
            
        buffer = self.crop(buffer, self.seq_len, self.crop_size)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer)
    
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

            buffer = buffer[time_index:time_index + clip_len,
                    height_index:height_index + crop_size,
                    width_index:width_index + crop_size, :]

        return buffer

# 0D dataset for generating probability curve
class DatasetFor0D(Dataset):
    def __init__(
        self, 
        ts_data : pd.DataFrame, 
        cols : List, 
        seq_len : int = 21, 
        dist:int = 3, 
        dt : float = 1.0 / 210 * 4,
        scaler : BaseEstimator = None,
        ):
        
        self.ts_data = ts_data
        self.seq_len = seq_len
        self.dt = dt
        self.cols = cols
        self.dist = dist
        self.indices = [idx for idx in range(0, len(self.ts_data) - seq_len - dist)]
        
        from sklearn.preprocessing import RobustScaler
        if scaler is None:
            self.scaler = RobustScaler()
        else:
            self.scaler = scaler
            
        self.ts_data[cols] = self.scaler.fit_transform(self.ts_data[cols].values)
   
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx : int):
        return self.get_data(idx)

    def get_data(self, idx : int):
        idx_srt = self.indices[idx]
        idx_end = idx_srt + self.seq_len
        
        data = self.ts_data[self.cols].iloc[idx_srt + 1: idx_end + 1].values
        data = torch.from_numpy(data)
        return data

class MultiModalDataset(Dataset):
    def __init__(
        self,
        root_dir : Optional[str],
        ts_data : pd.DataFrame, 
        ts_cols : List, 
        resize_height : Optional[int] = 256,
        resize_width : Optional[int] = 256,
        crop_size : Optional[int] = 224,
        frame_srt : int = 0,
        frame_end : int = -1,
        t_srt : int = 0,
        t_end : int = -1,
        vis_seq_len : int = 21, 
        ts_seq_len : int = 21,
        dist:int = 3, 
        dt:float = 1.0 / 210 * 4,
        scaler : Optional[BaseEstimator] = None,
        tau : int = 4,
        ):
        
        # properties of multi-modal data
        self.root_dir = root_dir
        self.ts_data = ts_data
        self.ts_cols = ts_cols
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.crop_size = crop_size
        self.vis_seq_len = vis_seq_len
        self.ts_seq_len = ts_seq_len
        self.tau = tau
        
        self.dist = dist
        self.dt = dt
        
        # get this value from disruption_list data
        self.frame_srt = frame_srt
        self.frame_end = frame_end
        self.t_srt = t_srt
        self.t_end = t_end
        
        # we have to focus on the disruptive phase
        # so, we will match the indices of video and ts data for backward direction
        # video path for showing performance -> Not used for prediction
        paths = glob2.glob(os.path.join(root_dir, "*"))
        
        # this is for showing performance : video path indices
        self.paths = sorted(paths)
        
        # video indices :  this will be used for prediction
        self.video_indices = None
        
        # ts data indices : tihs also will be used for prediction
        self.ts_indices = []
        
        # video file path list
        self.video_file_path = []
        
        # scaler for ts data
        if scaler is None:
            self.scaler = RobustScaler()
            self.ts_data[ts_cols] = self.scaler.fit_transform(self.ts_data[ts_cols].values)
        else:
            self.scaler = scaler
            self.ts_data[ts_cols] = self.scaler.transform(self.ts_data[ts_cols].values)
            
        # indice matching process
        video_indices = [i for i in reversed(range(frame_end, frame_srt, -tau))]
            
        # ts indices
        ts_idx_end = len(ts_data) - len(ts_data[ts_data.time > t_end])
        ts_idx_start = int(t_srt / self.dt)
        
        ts_indices = [i for i in reversed(range(ts_idx_end, ts_idx_start, -tau))]
        
        if len(video_indices) > len(ts_indices):
            video_indices = video_indices[- len(ts_indices):]
        elif len(video_indices) < len(ts_indices):
            ts_indices = ts_indices[-len(video_indices):]
            
        # self.ts_indices = ts_indices
        self.video_indices = video_indices
        
        # video file path
        for idx in video_indices:
            if idx > vis_seq_len * tau:
                self.video_file_path.append(self.paths[idx + 1 : idx - tau * vis_seq_len + 1 : -tau][::-1])  

        for idx in ts_indices:
            if idx > ts_seq_len * tau:
                self.ts_indices.append(idx)

        if len(self.video_file_path) > len(self.ts_indices):
            self.video_file_path = self.video_file_path[- len(ts_indices):]

        elif len(self.video_file_path) < len(self.ts_indices):
            self.ts_indices = self.ts_indices[-len(self.video_file_path):] 
                
    def __getitem__(self, idx : int):
        data_vis = self.get_video_data(idx)
        data_ts = self.get_ts_data(idx)
        return data_vis, data_ts
    
    def __len__(self):
        return len(self.ts_indices)

    def refill_temporal_slide(self, buffer:np.ndarray):
        for _ in range(self.vis_seq_len - buffer.shape[0]):
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

            buffer = buffer[time_index:time_index + clip_len,
                    height_index:height_index + crop_size,
                    width_index:width_index + crop_size, :]

        return buffer
    
    def load_frames(self, filepaths : List):
        buffer = np.empty((self.vis_seq_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, filepath in enumerate(filepaths):
            frame = np.array(cv2.imread(filepath)).astype(np.float32) 
            buffer[i] = frame
        return buffer
    
    def get_video_data(self, idx : int):
        buffer = self.load_frames(self.video_file_path[idx])
        if buffer.shape[0] < self.vis_seq_len:
            buffer = self.refill_temporal_slide(buffer)
        buffer = self.crop(buffer, self.vis_seq_len, self.crop_size)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer)

    def get_ts_data(self, idx : int):
        idx_end = self.ts_indices[idx]
        idx_srt = idx_end - self.ts_seq_len * self.tau
        data = self.ts_data[self.ts_cols].iloc[idx_srt + 1: idx_end + 1].values[::self.tau, :]
        data = torch.from_numpy(data).float()
        return data
    
def plot_exp_prob_type_1(ts_data_0D : pd.DataFrame, prob_list, time_x, shot_num : int, tftsrt, t_tq, t_cq, save_dir : str):
    
    # 0D parameters 
    t = ts_data_0D.time
    
    ip = ts_data_0D['\\RC03']
    iv = ts_data_0D['\\Iv']
    
    kappa = ts_data_0D['\\kappa']
    betap = ts_data_0D['\\BETAP_DLM03']
    li = ts_data_0D['\\li']
    q95 = ts_data_0D['\\q95']
    
    ne = ts_data_0D['\\ne_inter01']
    ne_ng_ratio = ts_data_0D['\\ne_nG_ratio']
    
    W_tot = ts_data_0D['\\WTOT_DLM03']
    
    te_core = ts_data_0D['\\TS_TE_CORE_AVG']
    te_edge = ts_data_0D['\\TS_TE_EDGE_AVG']
    ne_core = ts_data_0D['\\TS_NE_CORE_AVG']
    ne_edge = ts_data_0D['\\TS_NE_EDGE_AVG']
    
    # plot the disruption probability with plasma status
    fig = plt.figure(figsize = (21, 7))
    fig.suptitle("Disruption prediction with shot : {}".format(shot_num))
    gs = GridSpec(nrows = 3, ncols = 4)
    
    # kappa
    ax_kappa = fig.add_subplot(gs[0,0])
    ax_kappa.plot(t, kappa, c = 'b', label = 'kappa')
    ax_kappa.axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_kappa.axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_kappa.legend(loc = 'upper left', facecolor = 'white', framealpha=1)

    # betap
    ax_bp = fig.add_subplot(gs[1,0])
    ax_bp.plot(t, betap, c = 'b', label = 'betap')
    ax_bp.axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_bp.axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_bp.legend(loc = 'upper left', facecolor = 'white', framealpha=1)

    # internal inductance
    ax_li = fig.add_subplot(gs[2,0])
    ax_li.plot(t, li, c = 'b', label = 'li')
    ax_li.axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_li.axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_li.legend(loc = 'upper left', facecolor = 'white', framealpha=1)
    ax_li.set_xlabel("time(s)")
    
    # Te-core and Te-edge
    ax_te_core = fig.add_subplot(gs[0,1])
    ln_te_core = ax_te_core.plot(t, te_core, c = 'r', label = 'Te-core')
    ax_te_core.set_ylim([0, 10.0])
    ax_te_core.set_ylabel('', color = 'tab:red')
    ax_te_core.tick_params(axis='y', labelcolor='tab:red')
    
    ax_te_edge = ax_te_core.twinx()
    ln_te_edge = ax_te_edge.plot(t, te_edge, c = 'b', label = 'Te-edge')
    ax_te_edge.set_ylim([0, 5.0])
    ax_te_edge.set_ylabel('', color = 'tab:blue')
    ax_te_edge.tick_params(axis='y', labelcolor='tab:blue')
    ax_te_edge.axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_te_edge.axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    lns_te = ln_te_core + ln_te_edge
    labs_te = [l.get_label() for l in lns_te]
    ax_te_edge.legend(lns_te, labs_te, loc = 'upper left', facecolor = 'white', framealpha=1)

    # Ne-core and Ne-edge
    ax_ne_core = fig.add_subplot(gs[1,1])
    ln_ne_core = ax_ne_core.plot(t, ne_core, c = 'r', label = 'Ne-core')
    ax_ne_core.set_ylim([0, 4.0])
    ax_ne_core.set_ylabel('', color = 'tab:red')
    ax_ne_core.tick_params(axis='y', labelcolor='tab:red')
    
    ax_ne_edge = ax_ne_core.twinx()
    ln_ne_edge = ax_ne_edge.plot(t, ne_edge, c = 'b', label = 'Ne-edge')
    ax_ne_edge.set_ylim([0, 1.5])
    ax_ne_edge.set_ylabel('', color = 'tab:blue')
    ax_ne_edge.tick_params(axis='y', labelcolor='tab:blue')
    ax_ne_edge.axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ne_edge.axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    lns_ne = ln_ne_core + ln_ne_edge
    labs_ne = [l.get_label() for l in lns_ne]
    ax_ne_edge.legend(lns_ne, labs_ne, loc = 'upper left', facecolor = 'white', framealpha=1)
    
    # ne_ng_ratio
    ax_ne_ng = fig.add_subplot(gs[2,1])
    ax_ne_ng.plot(t, ne_ng_ratio, c = 'b', label = 'ne/ng')
    ax_ne_ng.axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ne_ng.axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_ne_ng.legend(loc = 'upper left', facecolor = 'white', framealpha=1)
    ax_ne_ng.set_xlabel("time(s)")
    
    # q95
    ax_q95 = fig.add_subplot(gs[0,2])
    ax_q95.plot(t, q95, c = 'b', label = 'q95')
    ax_q95.set_ylim([0,10])
    ax_q95.axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_q95.axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_q95.legend(loc = 'upper left', facecolor = 'white', framealpha=1)
    
    # plasma current and vessel current
    ax_Ip = fig.add_subplot(gs[1,2])
    ln_Ip = ax_Ip.plot(t, ip, c = 'r', label = 'Ip')
    ax_Ip.set_ylim([0.2, 0.8])
    ax_Ip.set_ylabel('', color = 'tab:red')
    ax_Ip.tick_params(axis='y', labelcolor='tab:red')
    
    ax_Iv = ax_Ip.twinx()
    ln_Iv = ax_Iv.plot(t, iv, c = 'b', label = 'Iv')
    ax_Iv.set_ylim([0, 0.15])
    ax_Iv.set_ylabel('', color = 'tab:blue')
    ax_Iv.tick_params(axis='y', labelcolor='tab:blue')
    ax_Iv.axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_Iv.axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    lns_Ip_Iv = ln_Ip + ln_Iv
    labs_Ip_Iv = [l.get_label() for l in lns_Ip_Iv]
    ax_Iv.legend(lns_Ip_Iv, labs_Ip_Iv, loc = 'upper left', facecolor = 'white', framealpha=1)

    # W_tot
    ax_w_tot = fig.add_subplot(gs[2,2])
    ax_w_tot.plot(t, W_tot, c = 'b', label = 'W-tot')
    ax_w_tot.axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_w_tot.axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_w_tot.legend(loc = 'upper left', facecolor = 'white', framealpha=1)
    ax_w_tot.set_xlabel("time(s)")
    
    # probability
    threshold_line = [0.5] * len(time_x)
    ax2 = fig.add_subplot(gs[:,3])
    ax2.plot(time_x, prob_list, 'b', label = 'disrupt prob')
    ax2.plot(time_x, threshold_line, 'k', label = "threshold(p = 0.5)")
    ax2.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed", label = "flattop (t={:.3f})".format(tftsrt))
    ax2.axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "TQ (t={:.3f})".format(t_tq))
    ax2.axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "CQ (t={:.3f})".format(t_cq))
    ax2.set_ylabel("probability")
    ax2.set_xlabel("time(unit : s)")
    ax2.set_ylim([0,1])
    ax2.set_xlim([0, max(time_x) + 0.05])
    ax2.legend(loc = 'upper left', facecolor = 'white', framealpha=1)
    
    fig.tight_layout()

    if save_dir:
        plt.savefig(save_dir, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)

    return fig

# function for ploting the probability curve for video network
def generate_prob_curve(
    file_path : str,
    model : torch.nn.Module, 
    device : str = "cpu", 
    save_dir : Optional[str] = "./results/real_time_disruption_prediction.gif",
    shot_list_dir : Optional[str] = "./dataset/KSTAR_Disruption_Shot_List_extend.csv",
    ts_data_dir : Optional[str] = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
    ts_cols : Optional[List] = None,
    shot_num : Optional[int] = None,
    clip_len : Optional[int] = None,
    dist_frame : Optional[int] = None,
    ):
    
    # obtain tTQend, tipmin and tftsrt
    shot_list_dir = pd.read_csv(shot_list_dir, encoding = "euc-kr")
    tTQend = shot_list_dir[shot_list_dir.shot == shot_num].tTQend.values[0]
    tftsrt = shot_list_dir[shot_list_dir.shot == shot_num].tftsrt.values[0]
    tipminf = shot_list_dir[shot_list_dir.shot == shot_num].tipminf.values[0]
    
    frame_srt = shot_list_dir[shot_list_dir.shot == shot_num].frame_startup.values[0]
    frame_end = shot_list_dir[shot_list_dir.shot == shot_num].frame_cutoff.values[0]

    # input data generation
    ts_data = pd.read_csv(ts_data_dir).reset_index()

    for col in ts_cols:
        ts_data[col] = ts_data[col].astype(np.float32)

    ts_data.interpolate(method = 'linear', limit_direction = 'forward')
    ts_data_0D = ts_data[ts_data['shot'] == shot_num]
    
    # video data
    dataset = VideoDataset(file_path, resize_height = 256, resize_width=256, crop_size = 128, seq_len = clip_len, dist = dist_frame, frame_srt=frame_srt, frame_end = frame_end)

    prob_list = []
    is_disruption = []

    model.to(device)
    model.eval()
    
    for idx in range(dataset.__len__()):
        with torch.no_grad():
            data = dataset.__getitem__(idx)
            output = model(data.to(device).unsqueeze(0))
            probs = torch.nn.functional.softmax(output, dim = 1)[:,0]
            probs = probs.cpu().detach().numpy().tolist()

            prob_list.extend(
                probs
            )
            
            is_disruption.extend(
                torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1].cpu().detach().numpy().tolist()
            )
            
    interval = 1
    fps = 210
    
    prob_list = [0] * (clip_len + frame_srt)+ prob_list
    
    # correction for startup peaking effect : we will soon solve this problem
    for idx, prob in enumerate(prob_list):
        
        if idx < fps * 1 and prob >= 0.5:
            prob_list[idx] = 0

    time_x = np.arange(dist_frame, len(prob_list) + dist_frame) * (1/fps) * interval
    
    print("\n(Info) flat-top : {:.3f}(s) | thermal quench : {:.3f}(s) | current quench : {:.3f}(s)\n".format(tftsrt, tTQend, tipminf))
    
    t_disrupt = tTQend
    t_current = tipminf
            
    fig = plot_exp_prob_type_1(ts_data_0D, prob_list, time_x, shot_num, tftsrt, tTQend, t_current, save_dir)

    return time_x, prob_list

def generate_prob_curve_from_0D(
    model : torch.nn.Module, 
    device : str = "cpu", 
    save_dir : Optional[str] = "./results/disruption_probs_curve.png",
    ts_data_dir : Optional[str] = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
    ts_cols : Optional[List] = None,
    shot_list_dir : Optional[str] = './dataset/KSTAR_Disruption_Shot_List_extend.csv',
    shot_num : Optional[int] = None,
    seq_len : Optional[int] = None,
    dist : Optional[int] = None,
    dt : Optional[int] = None,
    scaler : Optional[BaseEstimator] = None,
    ):
    
    # obtain tTQend, tipmin and tftsrt
    shot_list_dir = pd.read_csv(shot_list_dir, encoding = "euc-kr")
    tTQend = shot_list_dir[shot_list_dir.shot == shot_num].tTQend.values[0]
    tftsrt = shot_list_dir[shot_list_dir.shot == shot_num].tftsrt.values[0]
    tipminf = shot_list_dir[shot_list_dir.shot == shot_num].tipminf.values[0]
    
    frame_srt = shot_list_dir[shot_list_dir.shot == shot_num].frame_startup.values[0]
    frame_end = shot_list_dir[shot_list_dir.shot == shot_num].frame_cutoff.values[0]
    
    # input data generation
    ts_data = pd.read_csv(ts_data_dir).reset_index()

    for col in ts_cols:
        ts_data[col] = ts_data[col].astype(np.float32)

    ts_data.interpolate(method = 'linear', limit_direction = 'forward')
    
    ts_data_0D = ts_data[ts_data['shot'] == shot_num]
    ts_data_0D_before_scaling = ts_data_0D.copy(deep = True)
    
    t_start = ts_data_0D.time.values[0]
    
    # video data
    dataset = DatasetFor0D(ts_data_0D, ts_cols, seq_len, dist, dt, scaler)

    prob_list = []
    is_disruption = []

    model.to(device)
    model.eval()
    
    for idx in range(dataset.__len__()):
        with torch.no_grad():
            data = dataset.__getitem__(idx)
            output = model(data.to(device).unsqueeze(0))
            probs = torch.nn.functional.softmax(output, dim = 1)[:,0]
            probs = probs.cpu().detach().numpy().tolist()

            prob_list.extend(
                probs
            )
            
            is_disruption.extend(
                torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1].cpu().detach().numpy().tolist()
            )
            
    interval = 4
    fps = 210
    frame_srt = int(t_start * fps / interval)
    prob_list = [0] * (frame_srt + seq_len + dist) + prob_list + [0] * seq_len
    
    # correction for startup peaking effect : we will soon solve this problem
    for idx, prob in enumerate(prob_list):
        
        if idx < fps * 1 and prob >= 0.5:
            prob_list[idx] = 0
            
    from scipy.interpolate import interp1d
    
    prob_x = np.linspace(0, len(prob_list), num = len(prob_list), endpoint = True) * (interval/fps)
    prob_y = np.array(prob_list)
    f_prob = interp1d(prob_x, prob_y, kind = 'cubic')
    prob_list = f_prob(np.linspace(0, len(prob_list) * interval, num = len(prob_list) * interval, endpoint = True) * (1/fps))
    
    time_x = np.arange(0, len(prob_list)) * (1/fps)
    
    print("\n(Info) flat-top : {:.3f}(s) | thermal quench : {:.3f}(s) | current quench : {:.3f}(s)\n".format(tftsrt, tTQend, tipminf))
    
    t_disrupt = tTQend
    t_current = tipminf
    
    fig = plot_exp_prob_type_1(ts_data_0D_before_scaling, prob_list, time_x, shot_num, tftsrt, tTQend, t_current, save_dir)
    return time_x, prob_list

def generate_prob_curve_from_multi(
    file_path : str,
    model : torch.nn.Module, 
    device : str = "cpu", 
    save_dir : Optional[str] = "./results/disruption_probs_curve.png",
    ts_data_dir : Optional[str] = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
    ts_cols : Optional[List] = None,
    shot_list_dir : Optional[str] = './dataset/KSTAR_Disruption_Shot_List_extend.csv',
    shot_num : Optional[int] = None,
    vis_seq_len : Optional[int] = None,
    ts_seq_len : Optional[int] = None,
    dist : Optional[int] = None,
    dt : Optional[int] = None,
    scaler : Optional[BaseEstimator] = None,
    tau : int = 1,
    ):
    
    # obtain tTQend, tipmin and tftsrt
    shot_list_dir = pd.read_csv(shot_list_dir, encoding = "euc-kr")
    tTQend = shot_list_dir[shot_list_dir.shot == shot_num].tTQend.values[0]
    tftsrt = shot_list_dir[shot_list_dir.shot == shot_num].tftsrt.values[0]
    tipminf = shot_list_dir[shot_list_dir.shot == shot_num].tipminf.values[0]
    
    # define the start frame and end frame
    frame_srt = shot_list_dir[shot_list_dir.shot == shot_num].frame_startup.values[0]
    frame_end = shot_list_dir[shot_list_dir.shot == shot_num].frame_cutoff.values[0]
    
    # define the start time and end time
    t_srt = tftsrt
    t_end = tipminf

    # input data generation
    ts_data = pd.read_csv(ts_data_dir).reset_index()

    for col in ts_cols:
        ts_data[col] = ts_data[col].astype(np.float32)

    ts_data.interpolate(method = 'linear', limit_direction = 'forward')
    
    ts_data_0D = ts_data[ts_data['shot'] == shot_num]
    ts_data_0D_origin = ts_data_0D.copy(deep = True)
    
    # Multi-modal data
    dataset = MultiModalDataset(file_path, ts_data_0D, ts_cols, 256, 256, 128, frame_srt, frame_end, t_srt, t_end, vis_seq_len, ts_seq_len, dist, dt, scaler, tau)

    prob_list = []
    is_disruption = []

    model.to(device)
    model.eval()
    
    for idx in range(dataset.__len__()):
        with torch.no_grad():
            data_vis, data_ts = dataset.__getitem__(idx)
            data_vis = data_vis.to(device).unsqueeze(0)
            data_ts = data_ts.to(device).unsqueeze(0)
            output = model(data_vis, data_ts)
            probs = torch.nn.functional.softmax(output, dim = 1)[:,0]
            probs = probs.cpu().detach().numpy().tolist()

            prob_list.extend(
                probs
            )
            
            is_disruption.extend(
                torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1].cpu().detach().numpy().tolist()
            )
            
    t_srt = dataset.ts_data.iloc[dataset.ts_indices[0]].time.item()
    t_end = dataset.ts_data.iloc[dataset.ts_indices[-1]].time.item()
    
    dt_end = 1.0
    interval = tau
    fps = 210

    total_prob_list = [0] * int(t_srt * fps / interval + dist) + prob_list + [0] * int(dt_end * fps / interval)
    
    # correction for startup peaking effect : we will soon solve this problem
    for idx, prob in enumerate(total_prob_list):
        
        if idx < fps * 1.0 / interval and prob >= 0.5:
            total_prob_list[idx] = 0
            
    from scipy.interpolate import interp1d
    
    # there are different time interval between each region
    x_srt = [i * interval / fps for i in range(0, int(t_srt * fps / interval))]
    x_dist = [x_srt[-1] + i * 1 / fps for i in range(1, dist + 1)]
    x_prob_list = [x_dist[-1] + i * 1 / fps * interval for i in range(1, len(prob_list) + int(dt_end * fps / interval) + 1)]
    
    # prob_x and prob_y for interpolation
    prob_x = x_srt + x_dist + x_prob_list
    prob_x = np.array(prob_x) + dist * 1 / fps
    prob_y = np.array(total_prob_list)
    
    # interpolation for modifying the time interval
    f_prob = interp1d(prob_x, prob_y, kind = 'cubic', fill_value = "extrapolate")
    total_prob_list = f_prob(np.linspace(0, t_end + dt_end, num = len(total_prob_list) * interval, endpoint = True))
    
    # For convinent view, - dist added to move the graph left to the x-axis
    # time_x = np.arange(dist, len(total_prob_list) + dist) * (1/fps)
    time_x = np.linspace(0, t_end + dt_end, num = len(total_prob_list), endpoint = True)
    
    print("\n(Info) flat-top : {:.3f}(s) | thermal quench : {:.3f}(s) | current quench : {:.3f}(s)\n".format(tftsrt, tTQend, tipminf))
    
    t_disrupt = tTQend
    t_current = tipminf
    
    fig = plot_exp_prob_type_1(ts_data_0D_origin, total_prob_list, time_x, shot_num, tftsrt, tTQend, tipminf, save_dir)
    
    return time_x, prob_list

def plot_learning_curve(train_loss, valid_loss, train_f1, valid_f1, figsize : Tuple[int,int] = (12,6), save_dir : str = "./results/learning_curve.png"):
    x_epochs = range(1, len(train_loss) + 1)

    plt.figure(1, figsize=figsize, facecolor = 'white')
    plt.subplot(1,2,1)
    plt.plot(x_epochs, train_loss, 'ro-', label = "train loss")
    plt.plot(x_epochs, valid_loss, 'bo-', label = "valid loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("train and valid loss curve")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(x_epochs, train_f1, 'ro-', label = "train f1 score")
    plt.plot(x_epochs, valid_f1, 'bo-', label = "valid f1 score")
    plt.xlabel("epochs")
    plt.ylabel("f1 score")
    plt.title("train and valid f1 score curve")
    plt.legend()
    plt.savefig(save_dir)

def measure_computation_time(model : torch.nn.Module, input_shape : Tuple, n_samples : int = 1, device : str = "cpu"):
    model.to(device)
    model.eval()
    
    t_measures = []
    
    for n_iter in range(n_samples):
        
        torch.cuda.empty_cache()
        torch.cuda.init()
        
        with torch.no_grad():
            sample_data = torch.zeros(input_shape)
            t_start = time.time()
            sample_output = model(sample_data.to(device))
            t_end = time.time()
            dt = t_end - t_start
            t_measures.append(dt)
            
            sample_output.cpu()
            sample_data.cpu()
        
        del sample_data
        del sample_output
        
    # statistical summary
    dt_means = np.mean(t_measures)
    dt_std = np.std(t_measures)
    
    return dt_means, dt_std, t_measures    

def measure_computation_time_multi(model : torch.nn.Module, input_shape_vis : Tuple, input_shape_0D : Tuple, n_samples : int = 1, device : str = "cpu"):

    model.to(device)
    model.eval()
    
    t_measures = []
    
    for n_iter in range(n_samples):
        
        torch.cuda.empty_cache()
        torch.cuda.init()
        
        with torch.no_grad():
            sample_vis = torch.zeros(input_shape_vis)
            sample_0D = torch.zeros(input_shape_0D)
            
            t_start = time.time()
            sample_output = model(sample_vis.to(device), sample_0D.to(device))
            t_end = time.time()
            dt = t_end - t_start
            t_measures.append(dt)
            
            sample_output.cpu()
            sample_vis.cpu()
            sample_0D.cpu()

        del sample_output
        del sample_vis
        del sample_0D
        
    # statistical summary
    dt_means = np.mean(t_measures)
    dt_std = np.std(t_measures)
    
    return dt_means, dt_std, t_measures    