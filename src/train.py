from typing import Optional, List, Literal, Union
from src.loss import LDAMLoss, FocalLoss
import os
import pdb
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from src.evaluate import evaluate_tensorboard
from src.utils.EarlyStopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

# anomaly detection from training process : backward process
torch.autograd.set_detect_anomaly(True)

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
        # check that the batch_size = 1, if batch_size = 1, skip the process        
        if model_type == "single":
            if data.size()[0] <=1:
                continue
        else:
            data_video = data['video']
            if data_video.size()[0] <=1:
                continue
        
        # optimizer.zero_grad()
        # Efficient zero-out gradients
        for param in model.parameters():
            param.grad = None
        
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
            
        # if loss == nan, we have to break the training process
        # then, nan does not affect the weight of the parameters
        if not torch.isfinite(loss):
            print("train_per_epoch | Warning : loss nan occurs at batch_idx : {}".format(batch_idx))
            continue
        else:
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

    if total_size > 0:
        train_loss /= total_size
        train_acc /= total_size
        train_f1 = f1_score(total_label, total_pred, average = "macro")
        
    else:
        train_loss = 0
        train_acc = 0
        train_f1 = 0

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
            # optimizer.zero_grad()
            
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
    exp_dir : Optional[str] = None,
    max_norm_grad : Optional[float] = None,
    model_type : Literal["single","multi"] = "single",
    test_for_check_per_epoch : Optional[DataLoader] = None,
    is_early_stopping : bool = False,
    early_stopping_verbose : bool = True,
    early_stopping_patience : int = 12,
    early_stopping_delta : float = 1e-3
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
    if exp_dir:
        writer = SummaryWriter(exp_dir)
    else:
        writer = None
        
    if is_early_stopping:
        early_stopping = EarlyStopping(save_best_dir, early_stopping_patience, early_stopping_verbose, early_stopping_delta)
    else:
        early_stopping = None

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
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            
            writer.add_scalar('F1_score/train', train_f1, epoch)
            writer.add_scalar('F1_score/valid', valid_f1, epoch)
        
        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train acc : {:.3f}, valid acc : {:.3f}, train f1 : {:.3f}, valid f1 : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_acc, valid_acc, train_f1, valid_f1
                ))
                
                if test_for_check_per_epoch and writer is not None:
                    model.eval()
                    fig = evaluate_tensorboard(test_for_check_per_epoch, model, optimizer, loss_fn, device, 0.5, model_type)
                    writer.add_figure('Model-performance', fig, epoch)
                    model.train()
                    
        # save the last parameters
        torch.save(model.state_dict(), save_last_dir)

        # save the best parameters
        if  best_f1 < valid_f1:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch

        if early_stopping:
            early_stopping(valid_f1, model)
            if early_stopping.early_stop:
                print("Early stopping | epoch : {}, best f1 score : {:.3f}".format(epoch, best_f1))
                break
        else:
            torch.save(model.state_dict(), save_best_dir)

    # print("\n============ Report ==============\n")
    print("training process finished, best loss : {:.3f} and best acc : {:.3f}, best f1 : {:.3f}, best epoch : {}".format(
        best_loss, best_acc, best_f1, best_epoch
    ))
    
    if writer:
        writer.close()

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
    cls_num_list : Optional[List] = None,
    betas : List = [0, 0.25, 0.75, 0.9],
    model_type : Literal['single','multi'] = 'single',
    test_for_check_per_epoch : Optional[DataLoader] = None,
    is_early_stopping : bool = False,
    early_stopping_verbose : bool = True,
    early_stopping_patience : int = 12,
    early_stopping_delta : float = 1e-3
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
        
        if idx >= len(betas):
            idx = len(betas) - 1
        
        beta = betas[idx]
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        return per_cls_weights    
    
    # tensorboard
    if exp_dir:
        writer = SummaryWriter(exp_dir)
    else:
        writer = None
        
    if is_early_stopping:
        early_stopping = EarlyStopping(save_best_dir, early_stopping_patience, early_stopping_verbose, early_stopping_delta)
    else:
        early_stopping = None
    
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
                
                if test_for_check_per_epoch and writer is not None:
                    model.eval()
                    fig = evaluate_tensorboard(test_for_check_per_epoch, model, optimizer, loss_fn, device, 0.5, model_type)
                    writer.add_figure('Model-performance', fig, epoch)
                    model.train()

        # save the last parameters
        torch.save(model.state_dict(), save_last_dir)

        # save the best parameters
        if  best_f1 < valid_f1:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch

        if early_stopping:
            early_stopping(valid_f1, model)
            if early_stopping.early_stop:
                print("Early stopping | epoch : {}, best f1 score : {:.3f}".format(epoch, best_f1))
                break
        else:
            torch.save(model.state_dict(), save_best_dir)

    print("training process finished, best loss : {:.3f}, best acc : {:.3f}, best f1 : {:.3f}, best epoch : {}".format(
        best_loss, best_acc, best_f1, best_epoch
    ))
    
    if writer:
        writer.close()

    return  train_loss_list, train_acc_list, train_f1_list,  valid_loss_list, valid_acc_list, valid_f1_list