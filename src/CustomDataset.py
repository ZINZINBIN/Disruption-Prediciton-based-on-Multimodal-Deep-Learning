import os
import numpy as np
import pandas as pd
import torch
import random, glob2, cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, Dict, List, Union, Literal

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
DEFAULT_TS_COLS = [
    '\\q95', '\\ipmhd', '\\kappa', 
    '\\tritop', '\\tribot','\\betap','\\betan',
    '\\li', '\\WTOT_DLM03'
]

# Dataset for video model
class DatasetForVideo(Dataset):
    def __init__(
        self, 
        root_dir : Optional[str] = "./dataset/dur21_dis0",
        task : Literal["train", "valid", "test"] = "train", 
        augmentation : Optional[bool] = True, 
        augmentation_args : Optional[Dict] = None,
        resize_height : Optional[int] = 256,
        resize_width : Optional[int] = 256,
        crop_size : Optional[int] = 224,
        seq_len : int = 21,
        ):

        self.root_dir = root_dir # video root directory
        self.task = task # task : train / valid / test 
        self.augmentation = augmentation # video sequence augmentation
        self.augmentation_args = augmentation_args # argument for augmentation
        
        # resize each frame from video
        self.resize_height = resize_height
        self.resize_width = resize_width
        
        # crop
        self.crop_size = crop_size
        
        # video sequence length
        # warning : 0D data and video data should have equal sequence length
        self.seq_len = seq_len
        
        # data augmentation setup
        if augmentation_args is None:
            if self.augmentation is True:  
                self.augmentation_args = DEFAULT_AUGMENTATION_ARGS
            else:
                self.augmentation_args = None
        else:
            self.augmentation_args = augmentation_args

        # video_file_path : video file path : {database}/{shot_num}_{frame_start}_{frame_end}.avi
        # indices : index for tabular data, shot == shot_num, index <- df[df.frame_idx == frame_start].index
        self.video_file_path = []
        self.indices = []
        self.labels = []

        self.folder = os.path.join(self.root_dir, task)
        self.class_list = sorted(os.listdir(self.folder))
        self.n_classes = len(self.class_list)

        for cls in self.class_list:
            for fname in os.listdir(os.path.join(self.folder, cls)):
                self.video_file_path.append(os.path.join(self.folder, cls, fname))
                self.labels.append(cls)
        assert len(self.labels) == len(self.video_file_path), "video data and labels are not matched"

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(self.class_list)}
        self.index2label = {index: label for index, label in enumerate(self.class_list)}
        
        # Convert the list of label names into an array of label indices
        self.labels = np.array([self.label2index[label] for label in self.labels], dtype=int)

    def load_frames(self, file_dir : str):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float32)
            buffer[i] = frame
        return buffer
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx : int):

        label = torch.from_numpy(np.array(self.labels[idx]))
        return self.get_video_data(idx), label

    def get_video_data(self, index : int):
        buffer = self.load_frames(self.video_file_path[index])
        
        if buffer.shape[0] < self.seq_len:
            buffer = self.refill_temporal_slide(buffer)

        buffer = self.crop(buffer, self.seq_len, self.crop_size)

        if self.task == "train" and self.augmentation:
            buffer = self.brightness(buffer, val = self.augmentation_args["bright_val"], p = self.augmentation_args["bright_p"])
            buffer = self.contrast(buffer, self.augmentation_args["contrast_min"], self.augmentation_args["contrast_max"], p = self.augmentation_args["contrast_p"])
            buffer = self.blur(buffer, p = self.augmentation_args["blur_p"], kernel_size = self.augmentation_args["blur_k"])
            buffer = self.randomflip(buffer, p = self.augmentation_args["flip_p"])
            buffer = self.vertical_shift(buffer, ratio = self.augmentation_args["vertical_ratio"], p = self.augmentation_args["vertical_p"])
            buffer = self.horizontal_shift(buffer, ratio = self.augmentation_args["horizontal_ratio"], p = self.augmentation_args["horizontal_p"])

        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer)

    def refill_temporal_slide(self, buffer:np.ndarray):
        # if temporal length of buffer is not enought to clip len due to data leakage
        # copy some nearby data 
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

                    #frame = frame[:,:-to_shift, :]
                    #buffer[i] = cv2.resize(frame, dsize = (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
            else:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[:,-to_shift:, :] = frame[:,-to_shift:, :]
                    buffer[i] = ref

                    #frame = frame[:,-to_shift:, :]
                    #buffer[i] = cv2.resize(frame, dsize = (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)

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
                    #frame = frame[:-to_shift, :, :]
                    #buffer[i] = cv2.resize(frame, dsize = (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
                    
            else:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[-to_shift:, :, :] = frame[-to_shift:, :, :]
                    buffer[i] = ref

                    #frame = frame[-to_shift:, :, :]
                    #buffer[i] = cv2.resize(frame, dsize = (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
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

            # Crop and jitter the video using indexing. The spatial crop is performed on
            # the entire array, so each frame is cropped in the same location. The temporal
            # jitter takes place via the selection of consecutive frames
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
    
# Dataset for video model
class DatasetForVideo2(Dataset):
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
        
        self.labels = np.array(self.labels, dtype=int)
        self.n_classes = 2
    
        print("disrupt : {}, non-disrupt : {}".format(np.sum(self.labels==0), np.sum(self.labels == 1)))

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
    def __init__(self, ts_data : pd.DataFrame, disrupt_data : pd.DataFrame, seq_len : int = 21, cols : List = DEFAULT_TS_COLS, dist:int = 3, dt : float = 1.0 / 210 * 4):
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
        
        # ignore shot which have too many nan values
        shot_ignore = []
        for shot in tqdm(shot_list, desc = 'extract the null data'):
            df_shot = self.ts_data[self.ts_data.shot == shot]
            null_check = df_shot[self.cols].isna().sum()
            
            for c in null_check:
                if c > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    break
          
        shot_list = [shot_num for shot_num in shot_list if shot_num not in shot_ignore]

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
                    labels.append(1)
                    idx += self.seq_len // 3

                elif t > t_disrupt - self.dt * (self.seq_len + self.dist) and t <= t_disrupt:
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(0)
                    idx += self.seq_len // 3
                
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