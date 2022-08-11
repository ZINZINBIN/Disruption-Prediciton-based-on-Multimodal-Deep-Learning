import os
import numpy as np
import pandas as pd
import torch
import random
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, Dict, List, Union, Literal

# video data augmentation arguments(default)
DEFAULT_AUGMENTATION_ARGS = {
    "bright_val" : 30,
    "bright_p" : 0.25,
    "contrast_min" : 1,
    "contrast_max" : 1.5,
    "contrast_p" : 0.25,
    "blur_k" : 5,
    "blur_p" : 0.25,
    "flip_p" : 0.25,
    "vertical_ratio" : 0.2,
    "vertical_p" : 0.25,
    "horizontal_ratio" : 0.2,
    "horizontal_p" : 0.25
}

# multi-modal dataset : video(or image sequence) + numerical variables(time series tabular data)
class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir : str = "./dataset/dur21_dis0",
        task : Literal["train", "valid", "test"] = "train", 
        ts_data : Optional[pd.DataFrame] = None,
        ts_cols : Optional[List] = None,
        augmentation : bool = True, 
        augmentation_args : Optional[Dict] = None,
        resize_height : int = 256,
        resize_width : int = 256,
        crop_size : int = 224,
        seq_len : int = 21,
        mode : Literal['video','tabular','multi-modal'] = "multi-modal"
        ):
        self.root_dir = root_dir
        self.task = task
        self.augmentation = augmentation
        self.augmentation_args = augmentation_args
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.mode = mode

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

        if ts_data is None and self.mode != "video":
            ts_data_path = "./dataset/KSTAR_Disruption_ts_data_extend.csv"
            if os.path.exists(ts_data_path):
                self.ts_data = pd.read_csv(ts_data_path)
            else:
                raise RuntimeError("ts_data is invalid, check the directory or input ts data as pd.DataFrame")
        elif self.mode != "video":
            self.ts_data = ts_data
        else:
            self.ts_data = None
        
        if ts_cols is None and self.mode != "video":
            self.ts_cols = [
                '\\q95', '\\ipmhd', '\\kappa', 
                '\\tritop', '\\tribot','\\betap','\\betan',
                '\\li', '\\ne_inter01', '\\WTOT_DLM03'
            ]
        else:
            self.ts_cols = ts_cols

        for cls in self.class_list:
            for fname in os.listdir(os.path.join(self.folder, cls)):
                self.video_file_path.append(os.path.join(self.folder, cls, fname))
                self.labels.append(cls)

        assert len(self.labels) == len(self.video_file_path), "video data and labels are not matched"

        if self.mode != "video":
            for video_path in tqdm(self.video_file_path, desc="index matching for tabular and video data"):
                video_filename = video_path.split('/')[-1]
                shot_num, frame_start, frame_end = int(video_filename.split('_')[0]), int(video_filename.split('_')[1]), int(video_filename.split('_')[2])
                ts_data_shot = self.ts_data[self.ts_data.shot == shot_num]
                idx = ts_data_shot[ts_data_shot.frame_idx == frame_start].index.item()
                self.indices.append(idx)
            assert len(self.indices) == len(self.video_file_path), "video data and tabular data are not matched"
        else:
            self.indices = None

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

        if self.mode == "video":
            return self.get_video_data(idx), label
        elif self.mode == "tabular":
            return self.get_tabular_data(idx), label
        else:
            return self.get_video_data(idx), self.get_tabular_data(idx), label

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
    
    def get_tabular_data(self, index : int):
        ts_idx = self.indices[index]
        data = self.ts_data[self.ts_cols].loc[ts_idx:ts_idx+self.seq_len-1].values
        return torch.from_numpy(data)

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

    def crop(self, buffer : Union[np.ndarray, torch.Tensor], clip_len : int, crop_size : int):
        # randomly select time index for temporal jittering
        if buffer.shape[0] < clip_len :
            time_index = np.random.randint(abs(buffer.shape[0] - clip_len))
        elif buffer.shape[0] == clip_len :
            time_index = 0
        else :
            # print("buffer.shape[0] : ", buffer.shape[0])
            time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    # function for imbalanced dataset
    # used for LDAM loss and re-weighting
    def get_img_num_per_cls(self):
        
        classes = np.unique(self.labels)
        self.num_per_cls_dict = dict()

        for cls in classes:
            num = np.sum(np.where(self.labels == cls, 1, 0))
            self.num_per_cls_dict[cls] = num
         
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

    test_data = CustomDataset(
        root_dir = "./dataset/dur21_dis0",
        task = 'train',
        ts_data = None,
        ts_cols = None,
        augmentation=False,
        augmentation_args=None,
        resize_height = 256,
        resize_width=256,
        crop_size = 224,
        seq_len = 21,
        mode = "multi-modal"
    )

    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    sample_buffer, sample_data, sample_label = next(iter(test_loader))

    print('sample_buffer : ', sample_buffer.size())
    print('sample_data : ', sample_data.size())
    print('sample_target : ', sample_label.size())