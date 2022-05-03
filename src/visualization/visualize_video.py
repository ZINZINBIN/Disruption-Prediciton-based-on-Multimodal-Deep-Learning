import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from src.utils.utility import crop, normalize

def img2video(dir : str, fps : int = 210, t_start : int = 0, t_end : int = -1):
    return None

def show_all_frame(
    dir : str, 
    shot_list_dir : Optional[str] = None,
    shot_number : Optional[int] = None,
    fps : int = 210, 
    t_start : Optional[float] = 0, 
    t_end : Optional[float] = None, 
    t_interval : Optional[float] = 3,
    crop_size : int = 112,
    resize_width : int = 171,
    resize_height : int = 128
    ):

    capture = cv2.VideoCapture(dir)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0
    channels = 3
    retaining = True

    if shot_list_dir and shot_number:
        shot_list = pd.read_csv(shot_list_dir)
        shot_info = shot_list[shot_list["shot"] == shot_number]
        t_disrupt = shot_info["tTQend"].values[0]
    else:
        t_disrupt = None

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

    if t_start is None:
        t_start = 0

    if t_end is None:
        t_end = len(buffer) / fps

    assert t_start < t_end, "t_start should be less than t_end"
    assert t_end < len(buffer) / fps + 1, "t_end is out of range"
    assert t_interval < t_end - t_start, "t_interval should be less than t_end - t_start"

    if t_disrupt is not None:
        t_end = t_disrupt + t_interval
 
    show_indices = range(int(t_start * fps), int(t_end * fps), int(t_interval * fps))

    # plot the image with (4 X N) array
    fig_width = 18
    fig_height = 0
    fig_cols = 4
    fig_rows = len(show_indices) // fig_cols

    if len(show_indices) % fig_cols > 0:
        fig_rows += 1

    fig_height = int(fig_rows * 4.5) + 4

    axes=[]
    fig = plt.figure(figsize = (fig_width, fig_height))

    for idx in range(len(show_indices)):
        frame_idx = show_indices[idx]
        frame = buffer[frame_idx, :, :, :]
        t = frame_idx / fps
        axes.append(fig.add_subplot(fig_rows, fig_cols, idx+1))

        if t_disrupt is not None and t >= t_disrupt:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s), disrupted"
        else:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s)"

        axes[-1].set_title(subplot_title)  
        plt.imshow(frame)

    fig.tight_layout()    
    plt.show()
    
    return None