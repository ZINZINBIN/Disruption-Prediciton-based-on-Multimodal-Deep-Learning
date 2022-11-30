from typing import Optional, List, Literal, Union
from src.loss import LDAMLoss, FocalLoss
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

def train_per_epoch(
    train_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    max_norm_grad : Optional[float] = None,
    model_type : Literal["single","multi","multi-GB"] = "single"
    ):

    model.train()
    model.to(device)

    train_loss = 0
    train_acc = 0

    total_pred = np.array([])
    total_label = np.array([])
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        if model_type == "single":
            data = data.to(device)
            output = model(data)
        elif model_type == "multi":
            data_video = data['video'].to(device)
            data_0D = data['0D'].to(device)
            output = model(data_video, data_0D)
        elif model_type == "multi-GB":
            data_video = data['video'].to(device)
            data_0D = data['0D'].to(device)
            output, output_vis, output_ts = model(data_video, data_0D)
            
        target = target.to(device)
        
        if model_type == 'multi-GB':
            loss = loss_fn(output, output_vis, output_ts, target)
        else:
            loss = loss_fn(output, target)

        loss.backward()

        # use gradient clipping
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)

        optimizer.step()

        train_loss += loss.item()

        pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
        train_acc += pred.eq(target.view_as(pred)).sum().item()
        total_size += pred.size(0) 
        
        total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
        total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))
        
    if scheduler:
        scheduler.step()

    train_loss /= total_size
    train_acc /= total_size

    train_f1 = f1_score(total_label, total_pred, average = "macro")

    return train_loss, train_acc, train_f1

def valid_per_epoch(
    valid_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    model_type : Literal["single","multi","multi-GB"] = "single"
    ):

    model.eval()
    model.to(device)
    valid_loss = 0
    valid_acc = 0

    total_pred = np.array([])
    total_label = np.array([])
    total_size = 0

    for batch_idx, (data, target) in enumerate(valid_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            
            if model_type == "single":
                data = data.to(device)
                output = model(data)
            elif model_type == "multi":
                data_video = data['video'].to(device)
                data_0D = data['0D'].to(device)
                output = model(data_video, data_0D)
            elif model_type == "multi-GB":
                data_video = data['video'].to(device)
                data_0D = data['0D'].to(device)
                output, output_vis, output_ts = model(data_video, data_0D)
                
            target = target.to(device)
            
            if model_type == 'multi-GB':
                loss = loss_fn(output, output_vis, output_ts, target)
            else:
                loss = loss_fn(output, target)
    
            valid_loss += loss.item()
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            valid_acc += pred.eq(target.view_as(pred)).sum().item()
            total_size += pred.size(0)

            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    valid_loss /= total_size
    valid_acc /= total_size

    valid_f1 = f1_score(total_label, total_pred, average = "macro")

    return valid_loss, valid_acc, valid_f1

def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : Union[torch.nn.CrossEntropyLoss, LDAMLoss, FocalLoss],
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_dir : str = "./weights/best.pt",
    save_last_dir : str = "./weights/last.pt",
    exp_dir : str = './results',
    max_norm_grad : Optional[float] = None,
    criteria : Literal["f1_score", "acc", "loss"] = "f1_score",
    model_type : Literal["single","multi"] = "single"
    ):

    train_loss_list = []
    valid_loss_list = []
    
    train_acc_list = []
    valid_acc_list = []

    train_f1_list = []
    valid_f1_list = []

    best_acc = 0
    best_epoch = 0
    best_f1 = 0
    best_loss = torch.inf
    
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    
    # tensorboard
    writer = SummaryWriter(exp_dir)

    for epoch in tqdm(range(num_epoch), desc = "training process"):

        train_loss, train_acc, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad,
            model_type
        )

        valid_loss, valid_acc, valid_f1 = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device,
            model_type
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)
        
        # tensorboard recording : loss and score
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        
        writer.add_scalar('F1_score/train', train_f1, epoch)
        writer.add_scalar('F1_score/valid', valid_f1, epoch)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train acc : {:.3f}, valid acc : {:.3f}, train f1 : {:.3f}, valid f1 : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_acc, valid_acc, train_f1, valid_f1
                ))

        # save the best parameters
        if criteria == "acc" and best_acc < valid_acc:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best_dir)
            
        elif criteria == "f1_score" and best_f1 < valid_f1:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best_dir)
            
        elif criteria == "loss" and best_loss > valid_loss:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best_dir)

        # save the last parameters
        torch.save(model.state_dict(), save_last_dir)

    # print("\n============ Report ==============\n")
    print("training process finished, best loss : {:.3f} and best acc : {:.3f}, best f1 : {:.3f}, best epoch : {}".format(
        best_loss, best_acc, best_f1, best_epoch
    ))

    return  train_loss_list, train_acc_list, train_f1_list,  valid_loss_list,  valid_acc_list, valid_f1_list

# Deferred Re-weighting with LDAM or Focal Loss
def train_DRW(
    train_loader : DataLoader,
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : Union[LDAMLoss, FocalLoss],
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : int = 1,
    save_best_dir : str = "./weights/best.pt",
    save_last_dir : str = "./weights/last.pt",
    exp_dir : str = './results',
    max_norm_grad : Optional[float] = None,
    criteria : Literal["f1_score", "acc", "loss"] = "f1_score",
    cls_num_list : Optional[List] = None,
    betas : List = [0, 0.25, 0.75, 0.9],
    model_type : Literal['single','multi'] = 'single'
    ):

    train_loss_list = []
    valid_loss_list = []

    train_f1_list = []
    valid_f1_list = []

    train_acc_list = []
    valid_acc_list = []

    best_f1 = 0
    best_acc = 0
    best_epoch = 0
    best_loss = torch.inf
    
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    # class per weight update
    def _update_per_cls_weights(epoch : int, betas : List, cls_num_list : List):
        idx = epoch // int(num_epoch / len(betas))
        beta = betas[idx]
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        return per_cls_weights    
    
    # tensorboard
    writer = SummaryWriter(exp_dir)  
    
    for epoch in tqdm(range(num_epoch), desc = "training process - Deferred Re-weighting"):

        per_cls_weights = _update_per_cls_weights(epoch, betas, cls_num_list)

        # FocalLoss / LDAMLoss update weight
        loss_fn.update_weight(per_cls_weights)

        train_loss, train_acc, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            None,
            loss_fn,
            device,
            max_norm_grad,
            model_type
        )

        valid_loss, valid_acc, valid_f1 = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device,
            model_type
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        
        # tensorboard recording : loss and score
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        
        writer.add_scalar('F1_score/train', train_f1, epoch)
        writer.add_scalar('F1_score/valid', valid_f1, epoch)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train f1 : {:.3f}, valid f1 : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_f1, valid_f1
                ))

        # save the best parameters
        if criteria == "f1_score" and best_f1 < valid_f1:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best_dir)
        elif criteria == "acc" and best_acc < valid_acc:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best_dir)
        elif criteria == "loss" and best_loss > valid_loss:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best_dir)

        # save the last parameters
        torch.save(model.state_dict(), save_last_dir)

    # print("\n============ Report ==============\n")
    print("training process finished, best loss : {:.3f}, best acc : {:.3f}, best f1 : {:.3f}, best epoch : {}".format(
        best_loss, best_acc, best_f1, best_epoch
    ))

    return  train_loss_list, train_acc_list, train_f1_list,  valid_loss_list, valid_acc_list, valid_f1_list