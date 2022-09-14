import pandas as pd
import numpy as np
import scipy as sp
import os
import argparse
import glob2
from typing import List
from tqdm.auto import tqdm

# parser
parser = argparse.ArgumentParser(description="generate dataframe for multi-modal dataset from video and numerical data")

# data directory
parser.add_argument("--seq_len", type = int, default = 21)
parser.add_argument("--n_fps", type = int, default = 4)
parser.add_argument("--fps", type = float, default = 210)

args = vars(parser.parse_args())

seq_len = args['seq_len']
n_fps = args['n_fps']
fps = args['fps']

def frame_calculator(time, fps=210, gap=0):
    frame_time = 1./fps
    frame_num = time/frame_time
    frame_num = frame_num + gap
    return round(frame_num)

def get_frame_indx_per_shot(
    shot : int, 
    ts_data : pd.DataFrame, 
    df_shot_list:pd.DataFrame,
    seq_len : int,
    n_fps : int = 4,
    fps : float = 210
    ):
    
    ts_data_shot = ts_data[ts_data.shot == shot]
    df_shot_info = df_shot_list[df_shot_list.shot == shot]
    
    frame_tftsrt = df_shot_info['tftsrt'].apply(frame_calculator, fps = fps).values.item()
    frame_cutoff = df_shot_info['frame_cutoff']
    frame_tTQend = df_shot_info['frame_tTQend']
    frame_tipminf = df_shot_info['frame_tipminf']
    
    tipminf = df_shot_info['tipminf']
    tTQend = df_shot_info['tTQend']
    
    
def get_dataset_from_path(video_path : List):
    
    task_list = []
    shot_list = []
    frame_start_list = []
    frame_end_list = []
    is_disrupt_list = []
    
    for path in tqdm(video_path, "get dataset from video path"):
        task = path.split("/")[-3]
        is_disrupt = True if path.split("/")[-2] == "disruption" else False
        file_name = path.split("/")[-1]
        shot, frame_start, frame_end = int(file_name.split("_")[0]), int(file_name.split("_")[1]), int(file_name.split("_")[2])
        
        shot_list.append(shot)
        frame_start_list.append(frame_start)
        frame_end_list.append(frame_end)
        is_disrupt_list.append(is_disrupt)
        task_list.append(task)
        
    df = pd.DataFrame(data = {
        "frame_start" : frame_start_list,
        "frame_end" : frame_end_list,
        "is_disrupt" : is_disrupt_list,
        "shot" : shot_list,
        "task" : task_list
    })
    
    df = df.sort_values(by = ["shot","frame_start"], axis=0).reset_index(drop = True)
    return df

def sync_video_0D_data(
    ts_data : pd.DataFrame, 
    video_log : pd.DataFrame,
    df_shot_list:pd.DataFrame,
    seq_len : int,
    n_fps : int = 4,
    fps : float = 210
    ):
    
    shot_list = np.unique(video_log.shot.values)
    
    for shot in shot_list:
            
        ts_data_shot = ts_data[ts_data.shot == shot]
        df_shot_info = df_shot_list[df_shot_list.shot == shot]
        
        frame_tftsrt = df_shot_info['tftsrt'].apply(frame_calculator, fps = fps).values.item()
        frame_cutoff = df_shot_info['frame_cutoff']
        frame_tTQend = df_shot_info['frame_tTQend']
        frame_tipminf = df_shot_info['frame_tipminf']
        
        tipminf = df_shot_info['tipminf']
        tTQend = df_shot_info['tTQend']
    
         
if __name__ == '__main__':
    
    # video path list
    video_path = "./dataset/dur84_dis4"
    video_path_list_train = glob2.glob(video_path + "/train/*/*")
    video_path_list_valid = glob2.glob(video_path + "/valid/*/*")
    video_path_list_test = glob2.glob(video_path + "/test/*/*")
    
    video_log_train = get_dataset_from_path(video_path_list_train)
    video_log_valid = get_dataset_from_path(video_path_list_valid)
    video_log_test = get_dataset_from_path(video_path_list_test)
    
    video_log = pd.concat([video_log_train, video_log_valid, video_log_test], axis = 0).reset_index(drop = True)
    
    # 0D data upload
    ts_data = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index()
    ts_data.interpolate(method = 'linear', limit_direction = 'forward')
    
    # shot list upload
    df_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_extend.csv', encoding = "euc-kr")
    
    print(ts_data[['time','frame_idx','shot']].head())
    print(df_shot_list.head())
    
    df_mult = sync_video_0D_data(ts_data, video_log, df_shot_list, seq_len, n_fps, fps)
    df_mult.to_csv("./dataset/KSTAR_Disruption_multi_data.csv", index = False)