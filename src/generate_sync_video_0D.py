import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import argparse
import glob2
from typing import List
from tqdm.auto import tqdm
from CustomDataset import DEFAULT_TS_COLS

# parser
parser = argparse.ArgumentParser(description="generate dataframe for multi-modal dataset from video and numerical data")

# data directory
parser.add_argument("--video_path", type = str, default = "./dataset/dur84_dis4")
parser.add_argument("--seq_len", type = int, default = 21)
parser.add_argument("--n_fps", type = int, default = 4)
parser.add_argument("--fps", type = float, default = 210)

args = vars(parser.parse_args())

seq_len = args['seq_len']
n_fps = args['n_fps']
fps = args['fps']
video_path = args['video_path']

def frame_calculator(time, fps=210, gap=0):
    frame_time = 1./fps
    frame_num = time/frame_time
    frame_num = frame_num + gap
    return round(frame_num)

def compute_t_from_frame_reverse(frame : int, t_end : float, frame_end : int, fps : float = 210):
    t = t_end - (frame_end - frame) * 1 / fps
    return t
    
def get_dataset_from_path(video_path : List):
    
    task_list = []
    shot_list = []
    frame_start_list = []
    frame_end_list = []
    is_disrupt_list = []
    path_list = []
    
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
        path_list.append(path)
        
    df = pd.DataFrame(data = {
        "frame_start" : frame_start_list,
        "frame_end" : frame_end_list,
        "is_disrupt" : is_disrupt_list,
        "shot" : shot_list,
        "task" : task_list,
        "path":path_list
    })
    
    df = df.sort_values(by = ["shot","frame_start"], axis=0).reset_index(drop = True)
    return df

def sync_video_0D_data(
    ts_data : pd.DataFrame, 
    video_log : pd.DataFrame,
    df_shot_list:pd.DataFrame,
    seq_len : int,
    n_fps : int = 4,
    fps : float = 210,
    ts_cols : List = DEFAULT_TS_COLS
    ):
    
    # 1 step : video frame index matching with real time
    df_mult = video_log.copy(...)
    t_start_list = []
    t_end_list = []
    
    for _, row in df_mult.iterrows():
        
        shot = row['shot']
        df_shot_info = df_shot_list[df_shot_list.shot == shot]
        
        frame_tTQend = df_shot_info['frame_tTQend']
        frame_tipminf = df_shot_info['frame_tipminf']
        
        tipminf = df_shot_info['tipminf']
        tTQend = df_shot_info['tTQend']
        
        # synchronize tipminf and frame_tipminf
        # Since frame_tipminf = frame_cutoff - 1, there are some mis-matching with time - frame relation
        # so it is necessary to re-match time index and frame number
        frame_start = row['frame_start']
        frame_end = row['frame_end']
        
        t_start = compute_t_from_frame_reverse(frame_start, tipminf, frame_tipminf, fps)
        t_end = compute_t_from_frame_reverse(frame_end, tipminf, frame_tipminf, fps)
        
        t_start_list.append(t_start)
        t_end_list.append(t_end)
        
    df_mult['t_start'] = np.array(t_start_list, dtype = np.float32)
    df_mult['t_end'] = np.array(t_end_list, dtype = np.float32)
    
    # 2 step : time interpolation with df_mult and ts_data
    shot_list = np.unique(df_mult.shot.values)
    ts_interpolate = pd.DataFrame()
    
    t_start_index = []
    t_end_index = []
    t_index = 0
    
    for shot in shot_list:
        df_shot = df_mult[df_mult.shot == shot]
        
        t = ts_data[ts_data.shot == shot].time.values.reshape(-1,)
        t_new = np.array([])
        for _, row in df_shot.iterrows():
            t_start = row['t_start']
            t_end = row['t_end']
            dt_new = np.arange(t_end, t_start, -n_fps * 1.0 / fps)[:seq_len][::-1]
            t_new = np.concatenate((t_new, dt_new))
            
            t_start_index.append(t_index)
            t_end_index.append(t_index + len(dt_new) - 1)
            t_index += len(dt_new)
            
        # t_new = np.arange(min(df_shot.t_start.values), max(df_shot.t_end.values) + n_fps * 1.0 / fps, n_fps * 1.0 / fps)

        ts_partial = {}
        ts_partial['time'] = t_new
        
        for col in ts_cols:
            data = ts_data[ts_data.shot == shot][col].values.reshape(-1,)
            interp = interp1d(t, data, fill_value = 'extrapolate')
            data_extend = interp(t_new).reshape(-1,)
            ts_partial[col] = data_extend
        
        ts_partial = pd.DataFrame(data = ts_partial)
        ts_interpolate = pd.concat([ts_interpolate, ts_partial], axis = 0).reset_index(drop = True)
    
    df_mult['t_start_index'] = np.array(t_start_index)
    df_mult['t_end_index'] = np.array(t_end_index)
    
    return df_mult, ts_interpolate
         
if __name__ == '__main__':
    
    # video path list
    video_path_list_train = glob2.glob(video_path + "/train/*/*")
    video_path_list_valid = glob2.glob(video_path + "/valid/*/*")
    video_path_list_test = glob2.glob(video_path + "/test/*/*")
    
    video_log_train = get_dataset_from_path(video_path_list_train)
    video_log_valid = get_dataset_from_path(video_path_list_valid)
    video_log_test = get_dataset_from_path(video_path_list_test)
    
    video_log = pd.concat([video_log_train, video_log_valid, video_log_test], axis = 0).reset_index(drop = True)
    video_log = video_log.sort_values(by = ["shot","frame_start"], axis=0).reset_index(drop = True)
    
    # 0D data upload
    ts_data = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index()
    ts_data.interpolate(method = 'linear', limit_direction = 'forward')
    
    # shot list upload
    df_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_extend.csv', encoding = "euc-kr")
    
    print(ts_data[['time','frame_idx','shot']].head())
    print(df_shot_list.head())
    print(video_log.head())
    
    df_mult, ts_interpolate = sync_video_0D_data(ts_data, video_log, df_shot_list, seq_len, n_fps, fps)
    print(df_mult.head())
    print(ts_interpolate.head())
    
    df_mult.to_csv("./dataset/KSTAR_Disruption_multi_data.csv", index = False)
    ts_interpolate.to_csv("./dataset/KSTAR_Disruption_ts_data_for_multi.csv", index = False)