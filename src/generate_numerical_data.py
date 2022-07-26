import pandas as pd
import numpy as np
import scipy as sp
from typing import List, Tuple
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

def ts_interpolate(df : pd.DataFrame, cols : List, df_disruption : pd.DataFrame):
    
    df_interpolate = pd.DataFrame()
    shot_list = np.unique(df.shot.values).tolist()

    for shot in tqdm(shot_list):
        # ts data with shot number = shot
        df_shot = df[df.shot == shot]
        dict_extend = {}
        t = df_shot.time.values.reshape(-1,)

        t_start = 0
        dt = 1.0 / 210
        t_end = max(t)

        # quench info
        tTQend = df_disruption[df_disruption.shot == shot].tTQend.values[0]
        tftsrt = df_disruption[df_disruption.shot == shot].tftsrt.values[0]
        tipminf = df_disruption[df_disruption.shot == shot].tipminf.values[0]

        if t_end < tTQend:
            t_end = tTQend + dt

        t_extend = np.arange(t_start, t_end + dt, 1.0 / 210)
        dict_extend['time'] = t_extend
        dict_extend['shot'] = [shot for _ in range(len(t_extend))]

        for col in cols:
            data = df_shot[col].values.reshape(-1,)
            interp = interp1d(t, data, fill_value = 'extrapolate')
            data_extend = interp(t_extend).reshape(-1,)

            dict_extend[col] = data_extend

        df_shot_extend = pd.DataFrame(data = dict_extend)
        df_interpolate = pd.concat([df_interpolate, df_shot_extend], axis = 0)

    return df_interpolate


if __name__ == "__main__":
    df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data.csv")
    df_disrupt = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List.csv")
    
    cols = df.columns[df.notna().any()].drop(['Unnamed: 0','shot','time']).tolist()

    df_extend = ts_interpolate(df, cols, df_disrupt)
    df_extend.to_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv")