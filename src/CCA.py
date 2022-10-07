from typing import Optional, List, Literal, Union
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

class DeepCCA(nn.Module):
    def __init__(
        self, 
        encoder_1 : nn.Module, 
        encoder_2 : nn.Module, 
        ):
        super(DeepCCA, self).__init__()
        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2
        
    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor):
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2
    
# CCA Loss
# reference : https://github.com/Michaelvll/DeepCCA/
class CCALoss(nn.Module):
    def __init__(self, output_dim : int, use_all_singular_values : bool):
        super(CCALoss, self).__init__()
        self.output_dim = output_dim
        self.use_all_singular_values = use_all_singular_values
        
        self.r1 = 1e-3
        self.r2 = 1e-3
        self.eps = 1e-6
    
    def forward(self, h1 : torch.Tensor, h2 : torch.Tensor):

        h1, h2 = h1.t(), h2.t()
        
        o1 = h1.size(0)
        o2 = h1.size(0)
        
        m = h1.size(1)
        
        h1_ = h1 - h1.mean(dim = 1).unsqueeze(1)
        h2_ = h2 - h2.mean(dim = 1).unsqueeze(1)
        
        sigma_h12 = 1.0 / (m-1) * torch.matmul(h1_, h2_.t())
        sigma_h11 = 1.0 / (m-1) * torch.matmul(h1_, h1_.t()) + self.r1 * torch.eye(o1, device = h1.device)
        sigma_h22 = 1.0 / (m-1) * torch.matmul(h2_, h2_.t()) + self.r2 * torch.eye(o2, device = h1.device)
        
        [D1,V1] = torch.symeig(sigma_h11, eigenvectors=True)
        [D2,V2] = torch.symeig(sigma_h22, eigenvectors=True)
        
        # For numerical stability, use torch.gt
        pos_idx1 = torch.gt(D1, self.eps).nonzero()[:,0]
        D1 = D1[pos_idx1]
        V1 = V1[:, pos_idx1]
        
        pos_idx2 = torch.gt(D2, self.eps).nonzero()[:,0]
        D2 = D2[pos_idx2]
        V2 = V2[:, pos_idx2]
        
        sigma_h11_root_inv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        sigma_h22_root_inv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(sigma_h11_root_inv, sigma_h12), sigma_h22_root_inv)
        
        if self.use_all_singular_values:
            corr = torch.trace(
                torch.sqrt(torch.matmul(Tval.t(), Tval))
            )
        else:
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (
                torch.eye(trace_TT.shape[0]).to(h1.device)*self.r1
            ))
            
            U,V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>self.eps, U, (torch.ones(U.shape).float()*self.eps).to(h1.device))
            U = U.topk(self.output_dim)[0]
            corr = torch.sum(torch.sqrt(U))
            
        return -corr

def _train_per_epoch(
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

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x1 = data['video'].to(device)
        x2 = data['0D'].to(device)
        x1,x2 = model(x1, x2)
        print("x1 : ", x1)
        print("x2 : ", x2)
        loss = loss_fn(x1,x2)
        
        print("loss : ", loss)
        print("loss grad : ", loss.grad)
        loss.backward()

        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)

        optimizer.step()

        train_loss += loss.item()
        
    if scheduler:
        scheduler.step()
        
    total_size = batch_idx + 1
    train_loss /= total_size

    return train_loss

def _valid_per_epoch(
    valid_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    model.to(device)
    valid_loss = 0

    for batch_idx, (data, target) in enumerate(valid_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            x1 = data['video'].to(device)
            x2 = data['0D'].to(device)
            x1,x2 = model(x1, x2)
            loss = loss_fn(x1,x2)
            
            valid_loss += loss.item()

    total_size = batch_idx + 1
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

    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            x1 = data['video'].to(device)
            x2 = data['0D'].to(device)
            x1,x2 = model(x1, x2)
            loss = loss_fn(x1,x2)
            
            test_loss += loss.item()

    total_size = batch_idx + 1
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

        train_loss = _train_per_epoch(train_loader,model,optimizer,loss_fn,scheduler,device,max_norm_grad)
        valid_loss = _valid_per_epoch(valid_loader,model,optimizer,loss_fn,device)

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