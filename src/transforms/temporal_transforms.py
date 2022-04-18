import random
import math

class LoopPadding(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, frame_indices):
        out = frame_indices
        
        for idx in out:
            if len(out) >= self.size:
                break
            out.append(idx)
        
        return out
    
class TemporalBeginCrop(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, frame_indices):
        out = frame_indices[:self.size]
        
        for idx in out:
            if len(out) >= self.size:
                break
            out.append(idx)
        
        return out
    
class TemporalCenterCrop(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, frame_indices):
        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))
        
        out = frame_indices[begin_index : end_index]
        
        for idx in out:
            if len(out) >= self.size:
                break
            out.append(idx)
        return out
    
class TemporalRandomCrop(object):
    def __init__(self, size, gamma_tau):
        self.size = size
        self.gamma_tau = gamma_tau
        
    def __call__(self, frame_indices, t_stride = 1, size = None):
        trunc = size if size is not None else self.size
        
        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        
        end_index = min(begin_index + self.size, len(frame_indices))
        
        out = frame_indices[begin_index : end_index:t_stride * self.gamma_tau]
        out = out[:trunc//self.gamma_tau]
        
        for idx in out:
            if len(out) >= trunc // self.gamma_tau:
                break
            out.append(idx)
            
        return out