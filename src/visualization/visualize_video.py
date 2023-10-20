import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from src.utils.utility import crop, normalize

def img2video(dir : str, fps : int = 210, t_start : int = 0, t_end : int = -1):
    return None

def show_all_frame(
    root_dir : Optional[str] = "./dataset/raw_videos/raw_videos/",
    shot_list_dir : Optional[str] = "./dataset/KSTAR_Disruption_Shot_List.csv",
    shot_number : Optional[int] = None,
    fps : int = 210, 
    t_start : Optional[float] = 0, 
    t_end : Optional[float] = None, 
    t_interval : Optional[float] = 3,
    crop_size : int = 224,
    resize_width : int = 256,
    resize_height : int = 256,
    ):

    video_dir = os.path.join(root_dir, "%06dtv01.avi"%shot_number)

    if not os.path.exists(video_dir):
        video_dir = os.path.join(root_dir, "%06dtv02.avi"%shot_number)

    capture = cv2.VideoCapture(video_dir)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0
    channels = 3
    retaining = True

    if shot_list_dir and shot_number:
        shot_list = pd.read_csv(shot_list_dir, encoding = "euc-kr")
        shot_info = shot_list[shot_list["shot"] == shot_number]
        t_disrupt = shot_info["tTQend"].values[0]
        tifminf = shot_info["tipminf"].values[0]
    else:
        t_disrupt = None
        tifminf = None

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
        if t_disrupt is not None:
            t_end = t_disrupt + t_interval * 4
        else:
            t_end = len(buffer) / fps

    if t_interval is None:
        t_interval = 1.0 / fps

    assert t_start < t_end, "t_start should be less than t_end"
    assert t_interval < t_end - t_start, "t_interval should be less than t_end - t_start"

    # t : -100ms modification
    # add 0.1 to t_start and t_end
    # fps : 210 (origin) but occurs some delay when the frame count is large
    if frame_count > (4 + 0.1) * 210:
        fps_r = 207
    else:
        fps_r = fps

    show_indices = range(round((t_start + 0.1) * fps_r), round((t_end + 0.1) * fps_r), round(t_interval * fps_r))

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

    is_tq = False
    is_cq = False

    for idx in range(len(show_indices)):
        frame_idx = int(show_indices[idx])
        frame = buffer[frame_idx, :, :, :]
        t = frame_idx / fps_r - 0.1
        axes.append(fig.add_subplot(fig_rows, fig_cols, idx+1))

        if t_disrupt is not None and t >= t_disrupt and not is_tq:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s), thermal quench occurs"
            is_tq = True

        elif t >= t_disrupt and is_cq:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s), disrupted"

        elif tifminf is not None and t >= tifminf and not is_cq:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s), current quench"
            is_cq = True
        
        elif t>=t_disrupt and t <= tifminf:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s), disruptive phase"

        else:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s)"

        axes[-1].set_title(subplot_title)  
        
        frame = frame / np.amax(frame)
        frame = np.clip(frame, 0, 1)
        plt.imshow(frame[:,:,::-1])

    fig.tight_layout()    
    plt.show()

    del buffer
    del capture
    del shot_info