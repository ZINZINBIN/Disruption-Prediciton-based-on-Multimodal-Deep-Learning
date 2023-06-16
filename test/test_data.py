import pytest, torch
import glob2, os, random
import pandas as pd
import numpy as np
from src.dataset import DatasetForVideo, DatasetFor0D, MultiModalDataset2
from src.utils.sampler import ImbalancedDatasetSampler
from sklearn.preprocessing import RobustScaler

def test_data_check():
    
    path_0D_data = "./dataset/KSTAR_Disruption_ts_data_extend.csv"
    path_disrupt = "./dataset/KSTAR_Disruption_Shot_List.csv"

    ts_data = pd.read_csv(path_0D_data).reset_index()
    kstar_shot_list = pd.read_csv(path_disrupt, encoding = "euc-kr")
    
    # nan interpolation
    ts_data.interpolate(method = 'linear', limit_direction = 'forward')
    
    ts_cols = [
        '\\q95', '\\ipmhd', '\\kappa', '\\tritop', '\\tribot',
        '\\betap','\\betan','\\li', '\\WTOT_DLM03','\\ne_inter01', 
        '\\TS_NE_CORE_AVG', '\\TS_TE_CORE_AVG'
    ]

    for col in ts_cols:
        ts_data[col] = ts_data[col].astype(np.float32)
    
    # check if there is no data
    assert len(ts_data) > 0 and len(kstar_shot_list) > 0
    
    # nan + inf check
    assert np.sum(np.isinf(ts_data.values)) == 0, "Nan value exist in ts data"
    
    scaler = RobustScaler()    
    scaler.fit(ts_data[ts_cols].values)
    
    # test for specific case : dist 3 and seq len 21, no augmentation
    data = DatasetFor0D(ts_data, kstar_shot_list, seq_len = 21, cols = ts_cols, dist = 3, dt = 4 * 1 / 210, scaler = scaler)
    
    # label distribution for LDAM / Focal Loss
    data.get_num_per_cls()
    cls_num_list = data.get_cls_num_list()
    sampler = ImbalancedDatasetSampler(data)
    dataloader = torch.utils.data.DataLoader(data, batch_size = 256, sampler=sampler, num_workers = 4, pin_memory=False)
    
    # check the class distribution : no disrupt data
    assert np.prod(cls_num_list), "There are some classes which have zero value ..!"
    
    for batch_idx, (data, target) in enumerate(dataloader):
        assert torch.isinf(data).sum() ==  0, "nan or inf value exist in input tensor..!"
        assert torch.isinf(target).sum() == 0, "nan or inf value exist in target tensor..!"
        assert torch.max(data).abs() < 1e6, "the max value of input tensor has anomality"
    
    
    ''' video data check from dataloader
    - check nan or inf value from torch.Tensor obtained from dataloader
    - check anomal value from torch.Tensor
    '''
    
    root_dir = "./dataset/temp"
    shot_list = glob2.glob(os.path.join(root_dir, "*"))
    shot_list = random.choices(shot_list, k = len(shot_list) // 32)
    
    df_disrupt = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_extend.csv")
    
    # test for specific case : dist 3 and seq len 21, no augmentation
    data = DatasetForVideo(shot_list, df_disrupt, augmentation = False, augmentation_args=None, crop_size = 128, seq_len = 20, dist = 3)
    
    # label distribution for LDAM / Focal Loss
    data.get_num_per_cls()
    cls_num_list = data.get_cls_num_list()
    sampler = ImbalancedDatasetSampler(data)
    dataloader = torch.utils.data.DataLoader(data, batch_size = 12, sampler=sampler, num_workers = 4, pin_memory=False)
    
    # check the class distribution : no disrupt data
    assert np.prod(cls_num_list), "There are some classes which have zero value ..!"
    
    for batch_idx, (data, target) in enumerate(dataloader):
        assert torch.isinf(data).sum() ==  0, "nan or inf value exist in input tensor..!"
        assert torch.isinf(target).sum() == 0, "nan or inf value exist in target tensor..!"
        assert torch.max(data).abs() < 1e6, "the max value of input tensor has anomality"