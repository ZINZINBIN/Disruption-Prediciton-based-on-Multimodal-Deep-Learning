import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os, cv2, glob2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Literal, Union, List

# prepare video data
# since video data is too heavy, load image from directory
def video2img(file_path : str, width : int = 256, height: int = 256, overwrite : bool = True, save_path : Optional[str] = None):
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
            cv2.imwrite(filename = os.path.join(save_path, video_filename, '%06d.jpg'%count), img = frame)
        count += 1
    
    capture.release()

# Custom dataset : used for video data or multi-modal data
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
    
    
TS_COLS = [
    '\\q95', '\\ipmhd', '\\kappa', 
    '\\tritop', '\\tribot','\\betap','\\betan',
    '\\li', '\\WTOT_DLM03'
]

# generate real time performance as gif
def generate_real_time_experiment(
    file_path : str,
    model : torch.nn.Module, 
    device : str = "cpu", 
    save_dir : Optional[str] = "./results/real_time_disruption_prediction.gif",
    shot_list_dir : Optional[str] = "./dataset/KSTAR_Disruption_Shot_List.csv",
    ts_data_dir : Optional[str] = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
    ts_cols : List = TS_COLS,
    shot_num : Optional[int] = None,
    clip_len : Optional[int] = None,
    dist_frame : Optional[int] = None,
    plot_freq : int = 100,
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
    
    t = ts_data_0D.time
    ip = ts_data_0D['\\ipmhd'] * (-1)
    kappa = ts_data_0D['\\kappa']
    betap = ts_data_0D['\\betap']
    betan = ts_data_0D['\\betan']
    li = ts_data_0D['\\li']
    Bc = ts_data_0D['\\bcentr']
    q95 = ts_data_0D['\\q95']
    tritop = ts_data_0D['\\tritop']
    tribot = ts_data_0D['\\tribot']
    W_tot = ts_data_0D['\\WTOT_DLM03']
    ne = ts_data_0D['\\ne_inter01']
    te = ts_data_0D['\\TS_CORE10:CORE10_TE']
    
    # video data
    dataset = VideoDataset(file_path, resize_height = 256, resize_width=256, crop_size = 128, seq_len = clip_len, dist = dist_frame, frame_srt=frame_srt, frame_end = frame_end)

    prob_list = []
    is_disruption = []

    model.to(device)
    model.eval()
    
    from tqdm.auto import tqdm
    
    for idx in tqdm(range(dataset.__len__())):
        with torch.no_grad():
            data = dataset.__getitem__(idx)
            data = data.to(device).unsqueeze(0)
            output = model(data)
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
    
    prob_list = [0] * clip_len + prob_list
    
    # correction for startup peaking effect : we will soon solve this problem
    for idx, prob in enumerate(prob_list):
        
        if idx < fps * 1 and prob >= 0.5:
            prob_list[idx] = 0


    # indices = range(0, len(prob_list), 21)
    
    # for good performance, we use slight index different where the disruption occurs
    idx_distance = 21
    idx_interval = 0
    indices = []
    
    for idx in range(0, len(prob_list)):
        
        if idx_interval > idx_distance:
            indices.append(idx)
            idx_interval = 1
        else:
            idx_interval += 1
        
        if idx > frame_end - int(1.4 * fps/10) and idx_distance > 0 and idx < frame_end: 
            idx_distance = 0
            
        elif idx > frame_end and idx_distance == 0:
            idx_distance = 21
    
    frame_indices = range(frame_srt, frame_end + fps)
    prob_indices = range(0, frame_end + fps - frame_srt)
    time_x = np.arange(dist_frame, len(prob_list) + fps + dist_frame) * (1/fps) * interval
    
    print("probability : ", len(prob_list))
    print("frame : ", len(frame_indices))
    print("thermal quench : ", tTQend)
    print("current quench: ", tipminf)
    
    t_disrupt = tTQend
    t_current = tipminf
    
    # generate gif file using animation
    fig, axes = plt.subplots(nrows = 1, ncols=2, figsize = (12,6))
    prob_points = axes[1].plot([],[], label = 'disrupt prob')[0]
    time_text = axes[1].text(0.1, 0.9, s = "", fontsize = 12, transform = axes[1].transAxes)
    
    threshold_line = [0.5] * len(time_x)
    axes[1].plot(time_x, threshold_line, 'k', label = "threshold(p = 0.5)")
    
    video_paths = dataset.original_path
    
    frame = cv2.imread(video_paths[frame_srt])
    axes[0].imshow(frame)
    
    axes[1].axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "thermal quench")
    axes[1].axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "current quench")
    
    axes[1].set_ylabel("probability")
    axes[1].set_xlabel("time(unit : s)")
    axes[1].set_ylim([0,1])
    axes[1].set_xlim([0,max(time_x)])
    axes[1].legend(loc = 'upper right')
    
    import time
    start_time = time.time()

    def replay(idx : int):
        
        frame_idx = frame_indices[idx]
        prob_idx = prob_indices[idx]
        
        prob_points.set_data(time_x[:idx], prob_list[:prob_idx])
        time_text.set_text("t={:.3f}".format(time_x[idx]))
        frame = cv2.imread(video_paths[frame_idx])
        axes[0].imshow(frame)
        
        if idx % int(len(prob_list) / 10) == 0:
            end_time = time.time()
            print("# convert to gif | {:.3f} percent complete | time : {:.3f}".format(frame_idx/(frame_end + fps)* 100, end_time - start_time))

    from matplotlib import animation
    
    ani = animation.FuncAnimation(fig, replay, frames = indices)
    writergif = animation.PillowWriter(fps = plot_freq)
    ani.save(save_dir, writergif)
    return 