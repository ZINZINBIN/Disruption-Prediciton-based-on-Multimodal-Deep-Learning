import numpy as np
import pandas as pd
from typing import List
from scipy.interpolate import interp1d

def get_point_profile(df_shot : pd.DataFrame, cols : List, t):
    tp = df_shot[df_shot.time == t][cols].values.reshape(-1,)
    return tp

def interpolate(radius : List, te : np.array, n_points : int):
    r_min = radius[0]
    r_max = radius[-1]
    
    r_new = np.linspace(r_min, r_max, n_points, endpoint=True)
    fn = interp1d(radius, te, kind = 'cubic')
    
    te_interpolate = fn(r_new)
    return r_new, te_interpolate

def get_profile(df_shot : pd.DataFrame, t:float, radius : List, cols_core : List, cols_edge : List, n_points : int = 128):
    core = get_point_profile(df_shot, cols_core, t)
    edge = get_point_profile(df_shot, cols_edge, t)
    pe = np.concatenate((core, edge[1:]))
    r, pe = interpolate(radius, pe, n_points)
    pe = np.clip(pe, a_min = 0.1, a_max = 1e2)
    return r, pe