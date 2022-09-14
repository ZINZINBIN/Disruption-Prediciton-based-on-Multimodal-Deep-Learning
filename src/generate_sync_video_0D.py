import pandas as pd
import numpy as np
import scipy as sp

NEW_DATA_PATH = "./dataset/"

def get_frame_indx_per_shot(
    shot : int, 
    ts_data : pd.DataFrame, 
    df_shot_list:pd.DataFrame,
    seq_len : int,
    n_fps : int
    ):
    
    ts_data_shot = ts_data[ts_data.shot == shot]
    df_shot_info = df_shot_list[df_shot_list.shot == shot]
    
    frame_cutoff = df_shot_info['frame_cutoff']
    frame_tTQend = df_shot_info['frame_tTQend']
    frame_tipminf = df_shot_info['frame_tipminf']
    
    tipminf = df_shot_info['tipminf']
    tTQend = df_shot_info['tTQend']
    
    
    
    
if __name__ == '__main__':
    
    # integrated dataframe for Video + 0D data
    df_mult = pd.DataFrame()
    
    # 0D data upload
    ts_data = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index()
    ts_data.interpolate(method = 'linear', limit_direction = 'forward')
    
    # shot list upload
    df_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_extend.csv', encoding = "euc-kr")
    
    print(ts_data[['time','frame_idx','shot']].head())
    
    print(df_shot_list.head())