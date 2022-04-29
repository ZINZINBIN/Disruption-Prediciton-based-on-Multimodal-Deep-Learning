import torch
import cv2
import os
import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

def preprocessing_video(file_path : str, width : int = 256, height: int = 256, overwrite : bool = True, save_path : Optional[str] = None):
    '''preprocessing_video : load video data by cv2 to save as resized image file(.jpg)
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

def crop(buffer, original_height, original_width, crop_size):
    height_index = np.random.randint(original_height - crop_size)
    width_index = np.random.randint(original_width - crop_size)

    buffer = buffer[:, height_index:height_index + crop_size, width_index:width_index + crop_size, :]
    return buffer

def normalize(buffer:np.ndarray):
    for i, frame in enumerate(buffer):
        frame -= np.array([[[90.0, 98.0, 102.0]]])
        buffer[i] = frame
    return buffer

def time_split(buffer:np.ndarray, clip_len : int):
    frame_count = buffer.shape[0]
    h = buffer.shape[1]
    w = buffer.shape[2]
    c = buffer.shape[3]

    batch_size = frame_count - clip_len + 1
    dataset = np.empty((batch_size, clip_len, h, w, c), dtype = np.float32)

    for idx in range(0, batch_size):
        t_start = idx
        t_end = idx + clip_len
        dataset[idx, :, :, :, :] = buffer[t_start : t_end, :, :, :]
    
    return dataset.transpose((0, 4, 1, 2, 3))

# generate video to input data
def video2tensor(
    dir : str, 
    channels : int = 3, 
    clip_len : int = 42, 
    crop_size : int = 112,
    resize_width : int = 171,
    resize_height : int = 128
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
    dataset = time_split(buffer, clip_len)
    dataset = torch.from_numpy(dataset)

    return dataset
    
# generate distribution probability curve(t vs prob)
def generate_prob_curve(
    dataset : torch.Tensor, 
    model : torch.nn.Module, 
    batch_size : int = 32, 
    device : str = "cpu", 
    save_dir : Optional[str] = "./results/disruption_probs_curve.png",
    shot_list_dir : Optional[str] = "./dataset/KSTAR_Disruption_Shot_List.csv",
    shot_number : Optional[int] = None,
    clip_len : Optional[int] = None,
    dist_frame : Optional[int] = None,
    ):
    prob_list = []
    is_disruption = []
    video_len = dataset.size(0)

    model.to(device)
    model.eval()

    if video_len >= batch_size:
        for idx in range(int(video_len / batch_size)):
            with torch.no_grad():
                idx_start = batch_size * idx
                idx_end = batch_size * (idx + 1)

                frames = dataset[idx_start : idx_end, :, :, :, :]
                frames = frames.to(device)
                output = model(frames)
                probs = torch.nn.functional.softmax(output, dim = 1)[:,0]
                prob_list.extend(
                    probs.cpu().detach().numpy().tolist()
                )
                is_disruption.extend(
                    torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1].cpu().detach().numpy().tolist()
                )

    else:
        with torch.no_grad():
            frames = dataset[:, :, :, :, :]
            frames = frames.to(device)
            probs = model(frames)
            probs = torch.nn.functional.softmax(probs, dim = 1)[:,0]
            prob_list.extend(
                probs.cpu().detach().numpy().tolist()
            )
            is_disruption.extend(
                    torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1].cpu().detach().numpy().tolist()
                )


    if shot_list_dir and shot_number:
        shot_list = pd.read_csv(shot_list_dir)
        shot_info = shot_list[shot_list["shot"] == shot_number]
    else:
        shot_list = None
        shot_info = None

    if shot_info is not None:
        t_disrupt = shot_info["tTQend"].values[0]
    else:
        t_disrupt = None

    # clip_len + distance만큼 외삽 진행
    prob_list = [0] * (clip_len + dist_frame) + prob_list

    if save_dir:
        fps = 210
        time_x = np.arange(1, len(prob_list) + 1) * (1/fps)
        threshold_line = [0.5] * len(time_x)

        plt.figure(figsize = (8,5))
        plt.plot(time_x, threshold_line, 'k', label = "threshold(p = 0.5)")
        plt.plot(time_x, prob_list, 'b-', label = "disruption probs")
        # plt.plot(time_x, is_disruption, "r", label = "disruption line(predict)")

        if t_disrupt is not None:
            plt.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")

        plt.ylabel("probability")
        plt.xlabel("time(unit : s)")
        plt.ylim([0,1])
        plt.xlim([0,max(time_x)])
        plt.legend()
        plt.savefig(save_dir)

    return prob_list