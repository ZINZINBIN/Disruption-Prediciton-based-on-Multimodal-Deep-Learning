''' pre-test for model
    test code based on pytest and torcheck to check whether the improper value or case can happen from training process
    In this code, we generally check the nan value, inf value, device allocation, parameter change and loss explosion
    Before training networks, use this code to check the validity of the model architectures or model training algorithms
    
    Reference
    - torcheck usage : https://towardsdatascience.com/testing-your-pytorch-models-with-torcheck-cb689ecbc08c
    - pytest usage : https://binux.tistory.com/47
'''

import pytest
import torch, torcheck
import glob2, os, random
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from src.dataset import DatasetForVideo2, DatasetFor0D, MultiModalDataset2
from src.models.R2Plus1D import R2Plus1DClassifier
from src.models.slowfast import SlowFast
from src.models.ViViT import ViViT
from src.models.ts_transformer import TStransformer
from src.models.CnnLSTM import CnnLSTM
from src.utils.sampler import ImbalancedDatasetSampler
from src.loss import LDAMLoss, FocalLoss
from src.models.resnet import Bottleneck3D
from src.train import train_per_epoch

@pytest.fixture
def test_device():
    if torch.cuda.device_count() > 0:
        device = "cuda:0"
    else:
        device = "cpu"
    return device

# gpu allocation check
def test_cuda_setup():
    # torch cuda available
    assert torch.cuda.is_available(), "torch.cuda must be available"
    
    # torch gpu device must not be zero
    assert torch.cuda.device_count() > 0
        
    # initialize torch cuda and clear cache memory
    torch.cuda.init()
    torch.cuda.empty_cache()

# vision model check
def test_vision_model(test_device):

    device = test_device
    root_dir = "./dataset/temp"
    shot_list = glob2.glob(os.path.join(root_dir, "*"))
    shot_list = random.choices(shot_list, k = len(shot_list) // 32)
    
    df_disrupt = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_extend.csv")
    
    # test for specific case : dist 3 and seq len 21, no augmentation
    data = DatasetForVideo2(shot_list, df_disrupt, augmentation = False, augmentation_args=None, crop_size = 128, seq_len = 20, dist = 3)
    
    # label distribution for LDAM / Focal Loss
    data.get_num_per_cls()
    cls_num_list = data.get_cls_num_list()
    sampler = ImbalancedDatasetSampler(data)
    dataloader = torch.utils.data.DataLoader(data, batch_size = 12, sampler=sampler, num_workers = 4, pin_memory=False)
    
    # check the class distribution : no disrupt data
    assert np.prod(cls_num_list), "There are some classes which have zero value ..!"
   
    # for re-sapmling
    per_cls_weights = 1.0 / np.array(cls_num_list)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
    # 1 epoch test
    loss_LDAM = LDAMLoss(cls_num_list, max_m = 0.5, s = 1.0, weight = per_cls_weights)
    loss_Focal = FocalLoss(weight = per_cls_weights, gamma = 2.0)
    loss_CE = torch.nn.CrossEntropyLoss(reduction = "mean", weight = per_cls_weights)
    
    model_cases = ["R2Plus1D", "SlowFast", "ViViT"]
    
    for case in model_cases:
        
        if case == "R2Plus1D":
            model = R2Plus1DClassifier(
                input_size  = (3,20,128, 128),
                num_classes = 2, 
                layer_sizes = [1,2,2,1],
                pretrained = False, 
                alpha = 1.0
            )       
        elif case == "SlowFast":
            model = SlowFast(
                input_shape = (3,20,128, 128),
                block = Bottleneck3D,
                layers = [1,2,2,1],
                alpha = 4,
                tau_fast = 1,
                num_classes = 2,
                alpha_elu = 1.0,
            )
        elif case == "ViViT":
            model = ViViT(
                image_size = 128,
                patch_size = 16,
                n_classes = 2,
                n_frames = 20,
                dim = 512,
                depth = 4,
                n_heads = 8,
                pool = "cls",
                in_channels = 3,
                d_head = 64,
                dropout = 0.25,
                embedd_dropout=0.25,
                scale_dim = 4,
                alpha = 1.0
            )
        else:
            ValueError("can not identify the model case ..!")

        model.to(device)
        
        # optimizer : AdamW fixed
        optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-4)

        # register optimizer
        torcheck.register(optimizer)
        
        # check the model parameter changing
        torcheck.add_module_changing_check(model, module_name=case)
        
        # check the range of the output
        # before softmax, the output of the model should not be in range of (0,1) => why?
        torcheck.add_module_output_range_check(
            model,
            output_range=(0, 1),
            negate_range=True,
        )
        
        # NaN should not occurs at all!
        torcheck.add_module_nan_check(model)
        
        # inf value should not occurs at all!
        torcheck.add_module_inf_check(model)
        
        # test for training process
        for loss in [loss_CE, loss_Focal, loss_LDAM]:
            loss, _, _ = train_per_epoch(
                dataloader,
                model,
                optimizer,
                None,
                loss_CE,
                device,
                1.0,
                "single"
            )
            
            assert not np.isinf(loss), "loss contains infinite value..!"
        
        model.cpu()

# time series model check
def test_ts_model(test_device):
    device = test_device
    df_disrupt = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_extend.csv")
    
    # preparing 0D data for use
    df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index()

    # nan interpolation
    df.interpolate(method = 'linear', limit_direction = 'forward')
    
    ts_cols = [
        '\\q95', '\\ipmhd', '\\kappa', '\\tritop', '\\tribot',
        '\\betap','\\betan','\\li', '\\WTOT_DLM03','\\ne_inter01', 
        '\\TS_NE_CORE_AVG', '\\TS_TE_CORE_AVG'
    ]

    for col in ts_cols:
        df[col] = df[col].astype(np.float32)
        
    shot_list = np.unique(df.shot.values)
    shot_list = random.choices(shot_list, k = len(shot_list) // 32)
    df_ = pd.DataFrame({})
    
    for shot in shot_list:
        df_ = pd.concat([df_, df[df.shot == shot]], axis = 0)
        
    df_ = df
    
    scaler = RobustScaler()    
    scaler.fit(df[ts_cols].values)
    
    # test for specific case : dist 3 and seq len 21, no augmentation
    data = DatasetFor0D(df, df_disrupt, seq_len = 21, cols = ts_cols, dist = 3, dt = 4 * 1 / 210, scaler = scaler)
    
    # label distribution for LDAM / Focal Loss
    data.get_num_per_cls()
    cls_num_list = data.get_cls_num_list()
    sampler = ImbalancedDatasetSampler(data)
    dataloader = torch.utils.data.DataLoader(data, batch_size = 256, sampler=sampler, num_workers = 4, pin_memory=False)
    
    # check the class distribution : no disrupt data
    assert np.prod(cls_num_list), "There are some classes which have zero value ..!"
   
    # for re-sapmling
    per_cls_weights = 1.0 / np.array(cls_num_list)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
    # 1 epoch test
    loss_LDAM = LDAMLoss(cls_num_list, max_m = 0.5, s = 1.0, weight = per_cls_weights)
    loss_Focal = FocalLoss(weight = per_cls_weights, gamma = 2.0)
    loss_CE = torch.nn.CrossEntropyLoss(reduction = "mean", weight = per_cls_weights)
    
    model_cases = ["TStransformer"]
    
    for case in model_cases:
        
        if case == "TStransformer":
            model = TStransformer(
                n_features=len(ts_cols),
                feature_dims = 128,
                max_len = 21,
                n_layers = 4,
                n_heads = 8,
                dim_feedforward=512,
                dropout = 0.25,
                cls_dims = 64,
                n_classes = 2
            )
        elif case == "CnnLSTM":
            model = CnnLSTM(
                seq_len = 21,
                n_features=len(ts_cols),
                conv_dim = 32,
                conv_kernel = 3,
                conv_stride=1,
                conv_padding=1,
                lstm_dim=64,
                n_layers=2,
                bidirectional=True,
                n_classes=2
            )    
        
        else:
            ValueError("can not identify the model case ..!")

        model.to(device)
        
        # optimizer : AdamW fixed
        optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-4)

        # register optimizer
        torcheck.register(optimizer)
        
        # check the model parameter changing
        torcheck.add_module_changing_check(model, module_name=case)
        
        # check the range of the output
        # before softmax, the output of the model should not be in range of (0,1) => why?
        torcheck.add_module_output_range_check(
            model,
            output_range=(0, 1),
            negate_range=True,
        )
        
        # NaN should not occurs at all!
        torcheck.add_module_nan_check(model)
        
        # inf value should not occurs at all!
        torcheck.add_module_inf_check(model)
        
        # test for training process
        for loss in [loss_CE, loss_Focal, loss_LDAM]:
            loss, _, _ = train_per_epoch(
                dataloader,
                model,
                optimizer,
                None,
                loss_CE,
                device,
                1.0,
                "single"
            )
            
            assert not np.isinf(loss), "loss contains infinite value..!"
        
        model.cpu()
        
# multi-modal model check
def test_multi_modal(test_device):
    device = test_device
    pass