# generate dataframe with frame_cutoff
# we assume that frame for current quench = frame for cut-off - 1 frame (almost 9ms)
# frame for current quench and thermal quech will be re-defined

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import cv2, gc, os, glob2
from src.utils.utility import crop, normalize
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import warnings

# remove warning
warnings.filterwarnings("ignore")

# consider tv01 and tv02.avi 
PATH = "./dataset/raw_videos/raw_videos/"
video_list = glob2.glob(PATH + "*.avi")
video_shot_list = []
for path in video_list:
    shot_num = path.split("/")[-1].split(".")[0][:-4]
    shot_num = int(shot_num)
    video_shot_list.append(shot_num)

import numpy as np
video_shot_list = np.unique(np.array(video_shot_list)).tolist()

print("total video number : ", len(video_shot_list))

# load dataframe
kstar_shot_df = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List.csv', encoding = "euc-kr")
kstar_shot_df.head()

# generate new shot df

year_list = []
tftsrt_list = []
tipminf_list = []
tTQend_list = []
dt_list = []
shot_list = []

for shot_num in video_shot_list:
    t_tqend = kstar_shot_df[kstar_shot_df.shot == shot_num]["tTQend"].values[0]
    t_ipmin = kstar_shot_df[kstar_shot_df.shot == shot_num]["tipminf"].values[0]
    t_ftsrt = kstar_shot_df[kstar_shot_df.shot == shot_num]["tftsrt"].values[0]
    year = kstar_shot_df[kstar_shot_df.shot == shot_num]["year"].values[0]
    dt = t_ipmin - t_tqend

    year_list.append(year)
    tftsrt_list.append(t_ftsrt)
    tipminf_list.append(t_ipmin)
    tTQend_list.append(t_tqend)
    dt_list.append(dt)
    shot_list.append(shot_num)

new_shot_df = pd.DataFrame(
    {
        "shot" : shot_list,
        "year" : year_list,
        "tftsrt" : tftsrt_list,
        "tipminf" : tipminf_list,
        "tTQend":tTQend_list,
        "dt" : dt_list
    }
)

new_shot_df.head()



# generate dataframe with frame_cutoff
# we assume that frame for current quench = frame for cut-off - 1 frame (almost 9ms)
# frame for current quench and thermal quech will be re-defined
save_dir = "./results/disruption_phase"
resize_height = 196
resize_width = 196
crop_size = 128
fps = 210
eps = 0.075

frame_cutoff_width = 210 * 0.25

new_shot_list = []
frame_startup_list = []
frame_cutoff_list = []
frame_thermal_quench_list = []
frame_current_quench_list = []

def check_startup(frame : np.ndarray, thres : float = eps):
    mean = np.mean(frame)
    if mean > thres:
        return True
    else:
        return False

def check_cutoff(frame : np.ndarray, thres : float = eps):
    mean = np.mean(frame)
    if mean < thres:
        return True
    else:
        return False

def norm(img : np.ndarray):
    img = img.astype(np.float32)
    img /= 255.
    return img

if not os.path.exists(save_dir):
    os.mkdir(os.path.join(save_dir))

for _, row in tqdm(new_shot_df.iterrows()):
    shot_num = row['shot']
    tTQend = row["tTQend"]
    tipminf = row["tipminf"]
    tftsrt = row['tftsrt']
    
    file_path = os.path.join(save_dir, "%06dtv01.png"%shot_num)
    video_dir = os.path.join("./dataset/raw_videos/raw_videos/", "%06dtv01.avi"%shot_num)

    if not os.path.exists(video_dir):
        file_path = os.path.join(save_dir, "%06dtv02.png"%shot_num)
        video_dir = os.path.join("./dataset/raw_videos/raw_videos/", "%06dtv02.avi"%shot_num)
        
    if not os.path.exists(video_dir):
            continue

    capture = cv2.VideoCapture(video_dir)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # filter : if frame_count is not comparative with the disruption log data, then ignore the shot
    frame_cutoff_estimate = int(tipminf * 210)
    
    if frame_count < frame_cutoff_estimate:
        continue
    
    count = 0
    channels = 3
    retaining = True

    buffer = np.empty((frame_count, resize_height, resize_width, channels), np.dtype('float32'))
    is_startup = False
    is_cutoff = False
    frame_startup = 0
    frame_cutoff = 0
    frame_srt = int(tftsrt * fps + 0.1 * fps)

    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        
        if check_startup(frame, eps) and (is_startup is False):
            is_startup = True
            frame_startup = count

        if frame is None:
            frame = np.zeros((resize_width, resize_height, channels))

        if (frame_height != resize_height) or (frame_width != resize_width):
            frame = cv2.resize(frame, (resize_width, resize_height))

        if count > frame_srt and check_cutoff(norm(frame), eps) and not is_cutoff and count > frame_cutoff_estimate - frame_cutoff_width:
            frame_cutoff = count
            is_cutoff = True

        buffer[count] = frame
        count += 1

    capture.release()

    frame_current_quench = frame_cutoff - 1
    dt = row['dt']
    frame_thermal_quench = frame_current_quench - int(dt * fps)
    
    if frame_thermal_quench < 0 or frame_cutoff < 210:
        continue
    
    buffer = crop(buffer, resize_height, resize_width, crop_size)
    buffer = normalize(buffer)

    frame_interval = max(round((frame_cutoff - frame_thermal_quench + 2) / 8), 1)
    frame_indices = range(frame_thermal_quench - 1, frame_cutoff + 1, frame_interval)
    t_list = [tTQend - 1/fps + i / fps for i in range(len(frame_indices))]
    
    frame_startup = int((t_ftsrt + 0.1)* fps)

    frame_current_quench_list.append(frame_current_quench)
    frame_thermal_quench_list.append(frame_thermal_quench)
    frame_cutoff_list.append(frame_cutoff)
    frame_startup_list.append(frame_startup)
    new_shot_list.append(shot_num)

    # plot the image with (4 X N) array
    fig_width = 18
    fig_height = 0
    fig_cols = 4
    fig_rows = 2

    if len(frame_indices) % fig_cols > 0 and len(frame_indices) > fig_rows * fig_cols:
        fig_rows += 1

    fig_height = int(fig_rows * 4.5) + 4

    axes=[]
    fig = plt.figure(figsize = (fig_width, fig_height), facecolor = 'white')

    is_tq = False
    is_iq = False
    is_cutoff = False
    t_disrupt = 0

    for idx, frame_idx in enumerate(frame_indices):
        frame = buffer[frame_idx, :, :, :]
        t = t_list[idx]
        axes.append(fig.add_subplot(fig_rows, fig_cols, idx+1))

        if frame_idx == frame_thermal_quench:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s), thermal quench"
        elif frame_idx == frame_current_quench:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s), current quench"
        elif frame_idx >= frame_cutoff:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s), disrupted"
        elif frame_idx > frame_thermal_quench and frame_idx < frame_current_quench:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s), disruptive phase"
        else:
            subplot_title = "frame at t : " + str(round(t,3)) + "(s)"
    
        axes[-1].set_title(subplot_title)  
        plt.imshow(frame)

    fig.tight_layout()    
    plt.savefig(file_path, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
    plt.close()

    del buffer
    del capture
    del fig

    gc.collect()
    
# revised shot log information
year_list = []
tftsrt_list = []
tipminf_list = []
tTQend_list = []
dt_list = []
shot_list = []

for shot_num in video_shot_list:
    t_tqend = kstar_shot_df[kstar_shot_df.shot == shot_num]["tTQend"].values[0]
    t_ipmin = kstar_shot_df[kstar_shot_df.shot == shot_num]["tipminf"].values[0]
    t_ftsrt = kstar_shot_df[kstar_shot_df.shot == shot_num]["tftsrt"].values[0]
    year = kstar_shot_df[kstar_shot_df.shot == shot_num]["year"].values[0]
    dt = t_ipmin - t_tqend
    
    if shot_num in new_shot_list:
        year_list.append(year)
        tftsrt_list.append(t_ftsrt)
        tipminf_list.append(t_ipmin)
        tTQend_list.append(t_tqend)
        dt_list.append(dt)
        shot_list.append(shot_num)

new_shot_df = pd.DataFrame(
    {
        "shot" : shot_list,
        "year" : year_list,
        "tftsrt" : tftsrt_list,
        "tipminf" : tipminf_list,
        "tTQend":tTQend_list,
        "dt" : dt_list,
        'frame_startup' : frame_startup_list,
        'frame_cutoff' : frame_cutoff_list,
        'frame_tTQend' : frame_thermal_quench_list,
        'frame_tipminf' : frame_current_quench_list
    }
)

new_shot_df.to_csv("./dataset/KSTAR_Disruption_Shot_List_extend.csv", index = False)
new_shot_df.head()