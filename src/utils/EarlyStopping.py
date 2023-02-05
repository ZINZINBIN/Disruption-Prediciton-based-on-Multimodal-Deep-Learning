import torch
import torch.nn as nn

class EarlyStopping:
    def __init__(self, path : str, patience : int = 8, verbose : bool = False, delta : float = 0):
        self.path = path
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        
        self.n_count = 0
        self.best_score = 0
        self.early_stop = False
    
    def __call__(self, val_score, model:nn.Module):
        
        if self.best_score is None:
            self.save_checkpoint(val_score, model)
            
        elif val_score < self.best_score + self.delta:
            self.n_count += 1
            print("EarlyStopping counter : {} out of {}".format(self.n_count, self.patience))
            
            if self.n_count >= self.patience:
                self.early_stop = True

        else:
            self.save_checkpoint(val_score, model)
            self.n_count = 0
            
    def save_checkpoint(self, val_score, model:nn.Module):
        if self.verbose:
            print("Best score increase :{:.3f} -> {:.3f}".format(self.best_score, val_score))
            
        torch.save(model.state_dict(), self.path)
        self.best_score = val_score