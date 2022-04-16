import logging
import math
import numpy as np
import logging
import torch
import os
from datetime import datetime
import psutil
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def check_nan_losses(loss):
    if math.isnan(loss):
        raise RuntimeError("ERROR : Got Nan losses : {}".format(datetime.now()))

def params_count(model : torch.nn.Module, ignore_bn : bool = True):
    '''Compute the number of parameter in model
    Args : 
        model(nn.Module) : model to count the parameters
    '''
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    
    else:
        count = 0
        for m in model.parameters():
            if not isinstance(m, torch.nn.BatchNorm3d):
                for p in model.parameters(recurse=False):
                    count += p.numel()
    return count

def gpu_mem_usage():
    '''compute the gpu memory usage for current device'''
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    
    return mem_usage_bytes / 1024 ** 3

def cpu_mem_usage():
    '''compute the cpu memory usage for the current device'''
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3
    return usage, total


def _get_model_analysis_input(cfg):
    input_shape = cfg.MODEL.DEFAULT_INPUT_SHAPE
    input_shape = (1, *input_shape)
    sample_inputs = torch.zeros(input_shape)
    return sample_inputs

def is_eval_epoch(cfg, cur_epoch, multigrid_schedule):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs
        cur_epoch (int): current epoch.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True
    if multigrid_schedule is not None:
        prev_epoch = 0
        for s in multigrid_schedule:
            if cur_epoch < s[-1]:
                period = max(
                    (s[-1] - prev_epoch) // cfg.MULTIGRID.EVAL_FREQ + 1, 1
                )
                return (s[-1] - 1 - cur_epoch) % period == 0
            prev_epoch = s[-1]

    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0
