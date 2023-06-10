import pandas as pd
import numpy as np
import warnings
from typing import List
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
from src.config import Config
from src.profile import get_profile

config = Config()

warnings.filterwarnings(action = 'ignore')

def ts_interpolate(df : pd.DataFrame, df_disruption : pd.DataFrame, dt : float = 0.025, n_points : int = 128, use_profile : bool = False):
    
    df_interpolate = pd.DataFrame()
    
    # nan interpolation
    df = df.interpolate(method = 'linear', limit_direction = 'forward')
    
    # inf, -inf to nan
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # TS data : Nan -> 0
    tompson_cols = []
    tompson_cols += config.TS_TE_CORE_COLS
    tompson_cols += config.TS_TE_EDGE_COLS
    tompson_cols += config.TS_NE_CORE_COLS
    tompson_cols += config.TS_NE_EDGE_COLS
    
    df[tompson_cols] = df[tompson_cols].fillna(0)
    df[config.TCI] = df[config.TCI].fillna(0)
    
    # scaling for Ne
    df[config.TS_NE_CORE_COLS] = df[config.TS_NE_CORE_COLS].apply(lambda x : x / (1e19))
    df[config.TS_NE_EDGE_COLS] = df[config.TS_NE_EDGE_COLS].apply(lambda x : x / (1e19))
    
    # scaling for Te
    df[config.TS_TE_CORE_COLS] = df[config.TS_TE_CORE_COLS].apply(lambda x : x / (1e3))
    df[config.TS_TE_EDGE_COLS] = df[config.TS_TE_EDGE_COLS].apply(lambda x : x / (1e3))
    
    
    def _bound(x, value : float = 1e2):
        return x if abs(x) < value else value * x / abs(x)
    
    for col in tompson_cols:
        df[col] = df[col].apply(lambda x : _bound(x, 1e2))
    
    # 0D parameters : should be positive
    for col in config.DEFAULT_COLS:
        if col == '\\ipmhd' or col == '\\bcentr':
            df[col] = df[col].abs().values
        else:
            df[col] = df[col].apply(lambda x : x if x > 0 else 0)
            
    # scaling for Ipmhd
    df['\\ipmhd'] = df['\\ipmhd'].apply(lambda x : x / 1e6)
        
    # TCI data
    for col in config.TCI:
       df[col] = df[col].apply(lambda x : x if x > 0 else 0)
       
    # HA scaling
    df[config.HA] = df[config.HA].apply(lambda x : x / (1e18))
    
    # RC scaling
    for col in config.RC:
        if col == "\\RC03":
            df[col] = df[col].apply(lambda x : x / 1e6)
        elif col == '\\VCM03':
            df[col] = df[col].apply(lambda x : x / 1e6)
        elif col == '\\RCPPU1' or col == '\\RCPPL1':
            df[col] = df[col].apply(lambda x : x / 1e6)
       
    shot_list = [shot_num for shot_num in np.unique(df.shot.values).tolist() if shot_num in np.unique(df_disruption.shot.values).tolist()]
        
    # total features
    total_cols = config.DEFAULT_COLS + config.LM + config.HCM + config.DL + config.LV + config.RC + config.TCI + config.HA + config.TS
    total_cols = [col for col in total_cols if col not in config.EXCEPT_COLS]
    
    # Missing value processing : limit the number of process
    for shot in tqdm(shot_list, desc = 'missing value processing per shot'):
        df[df.shot == shot][total_cols] = df[df.shot == shot][total_cols].fillna(method="ffill")

    # filtering the experiment : too many nan values for measurement or time length is too short
    shot_ignore = []
    for shot in tqdm(shot_list, desc = 'remove the invalid values'):
        # dataframe per shot
        df_shot = df[df.shot==shot]
        
        # addition : ne_inter01 is important to check disruption event
        if (df_shot['\\ne_inter01'].isnull().sum() > 0.5 * len(df_shot)) or (df_shot["\\ne_inter01"].max() - df_shot['\\ne_inter01'].min() < 1e-3): 
            shot_ignore.append(shot)
            print("shot : {} - ne_inter01 null data".format(shot))
            continue
        
        # time length of the experiment is too short : at least larger than 2(s)
        if df_shot.time.iloc[-1] - df_shot.time.iloc[0] < 2.0:
            shot_ignore.append(shot)
            print("shot : {} - time length issue".format(shot))
            continue
        
        # 1st filter : null data ignore
        is_null = False
        for col in total_cols:
            if df_shot[total_cols].isnull().sum()[col] > 0.5 * len(df_shot):
                shot_ignore.append(shot)
                print("shot : {} - null data ignore".format(shot))
                is_null = True
                break
        
        if is_null:
            continue
        
        # 2nd filter : measurement error
        for col in config.DEFAULT_COLS:
            # null data
            if np.sum(df_shot[col] == 0) > 0.5 * len(df_shot):
                shot_ignore.append(shot)
                print("shot : {} - measurement issue".format(shot))
                break

            # constant value
            if df_shot[col].max() - df_shot[col].min() < 1e-3:
                shot_ignore.append(shot)
                print("shot : {} - invalid / constant value issue".format(shot))
                break
            
    original_shot_list = shot_list
    shot_list = [x for x in shot_list if x not in shot_ignore]
    
    print("# of valid shot : {} | # of all shot : {}".format(len(shot_list), len(original_shot_list)))
    
    for shot in tqdm(shot_list, desc = 'interpolation process'):
        
        if shot not in df.shot.values:
            print("shot : {} not exist".format(shot))
            continue

        # ts data with shot number = shot
        df_shot = df[df.shot == shot]
        df_shot[total_cols] = df_shot[total_cols].fillna(method = 'ffill')
        
        # outlier replacement
        for col in cols:
            
            # plasma current -> pass
            if col == '\\ipmhd':
                continue
            
            q1 = df_shot[col].quantile(0.15)
            q3 = df_shot[col].quantile(0.85)
            
            IQR = q3 - q1
            whisker_width = 1.25      
            
            lower_whisker = q1 - whisker_width * IQR
            upper_whisker = q3 + whisker_width * IQR
            
            df_shot.loc[:,col] = np.where(df_shot[col]>upper_whisker, upper_whisker, np.where(df_shot[col]<lower_whisker,lower_whisker, df_shot[col]))
        
        dict_extend = {}
        t = df_shot.time.values.reshape(-1,)

        t_start = 0
        t_end = max(t)

        # quench info
        tTQend = df_disruption[df_disruption.shot == shot].t_tmq.values[0]
        tftsrt = df_disruption[df_disruption.shot == shot].t_flattop_start.values[0]
        tipminf = df_disruption[df_disruption.shot == shot].t_ip_min_fault.values[0]
        
        # valid shot selection
        if t_end < tftsrt:
            print("Invalid shot : {} - loss of data".format(shot))
            continue
        
        elif t_end < 2:
            print("Invalid shot : {} - operation time too short".format(shot))
            continue
        
        elif int((t_end - tftsrt) / (t[1] - t[0])) < 4:
            print("Invalid shot : {} - data too small".format(shot))
            continue
        
        t_start = tftsrt - dt * 4
        
        if t_end >= tipminf - dt * 8:
            t_end = tipminf + dt * 8
            
        elif t_end < tipminf - dt * 8:
            print("Invalid shot : {} - operation time too short".format(shot))
            continue
        
        t_extend = np.arange(t_start, t_end + dt, dt)
        dict_extend['time'] = t_extend
        dict_extend['shot'] = [shot for _ in range(len(t_extend))]

        for col in cols:
            data = df_shot[col].values.reshape(-1,)
            interp = interp1d(t, data, kind = 'cubic', fill_value = 'extrapolate')
            data_extend = interp(t_extend).reshape(-1,)
            dict_extend[col] = data_extend

        df_shot_extend = pd.DataFrame(data = dict_extend)
        df_interpolate = pd.concat([df_interpolate, df_shot_extend], axis = 0).reset_index(drop = True)
        
    # Feature engineering
    # Tompson diagnostics : core value estimation
    df_interpolate['\\TS_NE_CORE_AVG'] = df_interpolate[config.TS_NE_CORE_COLS].mean(axis = 1)
    df_interpolate['\\TS_NE_EDGE_AVG'] = df_interpolate[config.TS_NE_EDGE_COLS].mean(axis = 1)
    
    df_interpolate['\\TS_TE_CORE_AVG'] = df_interpolate[config.TS_TE_CORE_COLS].mean(axis = 1)
    df_interpolate['\\TS_TE_EDGE_AVG'] = df_interpolate[config.TS_TE_EDGE_COLS].mean(axis = 1)
    
    # Greenwald density and fraction
    import math
    df_interpolate['\\nG'] = df_interpolate['\\ipmhd'] / math.pi / df_interpolate['\\aminor'] ** 2
    df_interpolate['\\ne_nG_ratio'] = df_interpolate['\\ne_inter01'] / df_interpolate['\\nG'] * 0.1
    
    df_interpolate['shot'] = df_interpolate['shot'].astype(int)
    
    # negative value removal
    for col in config.DEFAULT_COLS:
        if col == "\\ipmhd":
            df_interpolate[col] = df_interpolate[col].abs().values
        else:
            df_interpolate[col] = df_interpolate[col].apply(lambda x : x if x > 0 else 0)
    
    for col in config.TCI:
       df_interpolate[col] = df_interpolate[col].apply(lambda x : x if x > 0 else 0)
    
    for col in tompson_cols:
        df_interpolate[col] = df_interpolate[col].apply(lambda x : x if x > 0 else 0)
        
    # specific case
    df_interpolate['\\WTOT_DLM03'] = df_interpolate['\\WTOT_DLM03'].apply(lambda x : x if x > 0 else 0)
    
    if use_profile:
        # profile data generation
        ne_profile = np.zeros((len(df_interpolate), n_points))
        te_profile = np.zeros((len(df_interpolate), n_points))
        idx = 0
        
        shot_list = np.unique(df_interpolate['shot'].values)
        
        for shot in tqdm(shot_list, desc = 'profile generation'):
            
            df_shot = df_interpolate[df_interpolate.shot == shot].copy()
            tes = []
            nes = []
            for t in df_shot.time:
                _, te = get_profile(df_shot, t, radius = config.RADIUS, cols_core = config.TS_TE_CORE_COLS, cols_edge = config.TS_TE_EDGE_COLS, n_points = n_points)
                _, ne = get_profile(df_shot, t, radius = config.RADIUS, cols_core = config.TS_NE_CORE_COLS, cols_edge = config.TS_NE_EDGE_COLS, n_points = n_points)
                tes.append(te.reshape(-1,1))
                nes.append(ne.reshape(-1,1))
                
            ne_profile[idx:idx + len(df_shot),:] = np.concatenate(nes, axis = 1).transpose(1,0)
            te_profile[idx:idx + len(df_shot),:] = np.concatenate(tes, axis = 1).transpose(1,0)
                
            idx += len(df_shot)    
        
        profile_info = {
            "te" : te_profile,
            "ne" : ne_profile
        }
    else:
        profile_info = None
    
    return df_interpolate, profile_info

if __name__ == "__main__":
    
    df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_revised.csv")
    df_disrupt = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_2022.csv", encoding = "euc-kr")    
    
    print(df.describe())

    cols = df.columns[df.notna().any()].drop(['Unnamed: 0','shot','time']).tolist()
    
    fps = 210
    
    # if 0D network only
    # dt = 1 / fps * 4
    
    # if multi-modal data
    dt = 1 / fps * 1
    
    n_points = 128
    
    df_extend, profile_info = ts_interpolate(df, df_disrupt, dt, n_points)
    df_extend['frame_idx'] = df_extend.time.apply(lambda x : int(round(x * fps)))
    
    # 0D dataset
    # df_extend.to_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv", index = False)
    
    # multimodal dataset
    df_extend.to_csv("./dataset/KSTAR_Disruption_ts_data_5ms.csv", index = False)
    
    if profile_info is not None:
        np.savez("./dataset/profiles.npz", te = profile_info['te'], ne = profile_info['ne'])
    
    print(df_extend.describe())