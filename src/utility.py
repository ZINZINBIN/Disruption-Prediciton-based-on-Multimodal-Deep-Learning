import torch
import cv2
import os
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
    