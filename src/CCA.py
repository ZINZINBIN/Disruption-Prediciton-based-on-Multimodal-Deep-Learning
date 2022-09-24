from typing import Optional, List, Literal, Union
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from src.loss import CCALoss

class DeepCCA(nn.Module):
    def __init__(
        self, 
        encoder_1 : nn.Module, 
        encoder_2 : nn.Module, 
        cca_loss : CCALoss,
        ):
        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2
        self.cca_loss = cca_loss
        
    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor):
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2
    
    def compute_loss(self, x1 : torch.Tensor, x2 : torch.Tensor):
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        loss = self.cca_loss(x1,x2)
        return loss
    
def train_per_epoch(
    train_loader : DataLoader, 
    model : DeepCCA,
    optimizer : torch.optim.Optimizer,
    loss_fn : CCALoss,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    device : str = "cpu",
    max_norm_grad : Optional[float] = None,
    ):

    model.train()
    model.to(device)

    train_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x1 = data['video'].to(device)
        x2 = data['0D'].to(device)
        x1,x2 = model(x1, x2)
        loss = loss_fn(x1,x2)
        loss.backward()

        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)

        optimizer.step()

        train_loss += loss.item()
        total_size += loss.size(0) 
        
    if scheduler:
        scheduler.step()

    train_loss /= total_size

    return train_loss

def valid_per_epoch(
    valid_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    model.to(device)
    valid_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(valid_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            x1 = data['video'].to(device)
            x2 = data['0D'].to(device)
            x1,x2 = model(x1, x2)
            loss = loss_fn(x1,x2)
            
            valid_loss += loss.item()
            total_size += loss.size(0) 

    valid_loss /= total_size

    return valid_loss

def evaluate_cca_loss(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):
    model.eval()
    model.to(device)
    test_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            x1 = data['video'].to(device)
            x2 = data['0D'].to(device)
            x1,x2 = model(x1, x2)
            loss = loss_fn(x1,x2)
            
            test_loss += loss.item()
            total_size += loss.size(0) 

    test_loss /= total_size
    return test_loss

def train_cca(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : DeepCCA,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : CCALoss,
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_dir : str = "./weights/cca_best.pt",
    save_last_dir : str = "./weights/cca_last.pt",
    max_norm_grad : Optional[float] = None,
    ):

    train_loss_list = []
    valid_loss_list = []

    best_epoch = 0
    best_loss = torch.inf

    for epoch in tqdm(range(num_epoch), desc = "training CCA process"):

        train_loss = train_per_epoch(train_loader,model,optimizer,loss_fn,scheduler,device,max_norm_grad)
        valid_loss = valid_per_epoch(valid_loader, model,optimizer,loss_fn,device)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}".format(epoch+1, train_loss, valid_loss))

        # save the best parameters
        if best_loss > valid_loss:
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best_dir)

        # save the last parameters
        torch.save(model.state_dict(), save_last_dir)

    print("(Report) training CCA process finished, best loss : {:.3f}, best epoch : {}".format(best_loss, best_epoch))

    return  train_loss_list, valid_loss_list