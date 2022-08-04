import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset

class ImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset : Dataset, indices = None, num_samples = None, callback_get_label = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.callback_get_label = callback_get_label
        
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        
        label_to_count = {}
        
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
                
            else:
                label_to_count[label] = 1
            
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.labels[idx]
    
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement = True
        ))
    
    def __len__(self):
        return self.num_samples
        