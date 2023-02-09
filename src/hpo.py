from typing import Optional, List, Literal, Union
from src.loss import LDAMLoss, FocalLoss, CELoss
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from src.utils.EarlyStopping import EarlyStopping
from ray import tune
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

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
        total_size += pred.size(0) 
        
        total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
        total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))
        
    if scheduler:
        scheduler.step()

    if total_size > 0:
        train_loss /= total_size
        train_f1 = f1_score(total_label, total_pred, average = "macro")
        
    else:
        train_loss = 0
        train_f1 = 0
    
    model.cpu()

    return train_loss, train_f1

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
            total_size += pred.size(0)

            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    valid_loss /= total_size
    valid_f1 = f1_score(total_label, total_pred, average = "macro")
    
    model.cpu()

    return valid_loss, valid_f1

def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : Union[torch.nn.CrossEntropyLoss, LDAMLoss, FocalLoss],
    device : str = "cpu",
    num_epoch : int = 64,
    max_norm_grad : Optional[float] = None,
    model_type : Literal["single","multi"] = "single",
    checkpoint_dir : Optional[str] = None
    ):

    train_loss_list = []
    valid_loss_list = []
    
    train_f1_list = []
    valid_f1_list = []

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in tqdm(range(num_epoch), desc = "training process for hyperparameter optimization"):

        train_loss, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad,
            model_type
        )

        valid_loss, valid_f1 = valid_per_epoch(
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
               
        # tune checkpoint and save
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # tune report
        tune.report(loss=valid_loss, f1_score=valid_f1)
        
    return  train_loss_list, train_f1_list,  valid_loss_list, valid_f1_list

# Deferred Re-weighting with LDAM or Focal Loss
def train_DRW(
    train_loader : DataLoader,
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : Union[LDAMLoss, FocalLoss],
    device : str = "cpu",
    num_epoch : int = 64,
    max_norm_grad : Optional[float] = None,
    cls_num_list : Optional[List] = None,
    betas : List = [0, 0.25, 0.75, 0.9],
    model_type : Literal['single','multi'] = 'single',
    checkpoint_dir : Optional[str] = None
    ):

    train_loss_list = []
    valid_loss_list = []

    train_f1_list = []
    valid_f1_list = []


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
        
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    for epoch in tqdm(range(num_epoch), desc = "training process(DRW) for hyperparameter optimization"):

        per_cls_weights = _update_per_cls_weights(epoch, betas, cls_num_list)

        # FocalLoss / LDAMLoss update weight
        loss_fn.update_weight(per_cls_weights)

        train_loss, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            None,
            loss_fn,
            device,
            max_norm_grad,
            model_type
        )

        valid_loss, valid_f1 = valid_per_epoch(
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
        
        # tune checkpoint and save
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # tune report
        tune.report(loss=valid_loss, f1_score=valid_f1)

    return  train_loss_list, train_f1_list,  valid_loss_list, valid_f1_list

def evaluate(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    loss_fn : Optional[torch.nn.Module]= None,
    device : Optional[str] = "cpu",
    threshold : float = 0.5,
    model_type : Literal["single","multi","multi-GB"] = "single"
    ):

    test_loss = 0
    test_f1 = 0
    total_pred = np.array([])
    total_label = np.array([])

    if device is None:
        device = torch.device("cuda:0")

    model.to(device)
    model.eval()

    total_size = 0

    for idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            
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
    
            test_loss += loss.item()
            # pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            
            pred = torch.nn.functional.softmax(output, dim = 1)[:,0]
            pred = torch.logical_not((pred > torch.FloatTensor([threshold]).to(device)))
            total_size += pred.size(0)
            
            pred_normal = torch.nn.functional.softmax(output, dim = 1)[:,1].detach()
            
            total_pred = np.concatenate((total_pred, pred_normal.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))
            
    test_loss /= (idx + 1)
    total_pred = np.nan_to_num(total_pred, copy = True, nan = 0, posinf = 1.0, neginf = 0)
    lr_probs = total_pred
    total_pred = np.where(total_pred > 1 - threshold, 1, 0)
    
    # f1 score
    test_f1 = f1_score(total_label, total_pred, average = "macro")
    
    # auc score
    test_auc = roc_auc_score(total_label, total_pred, average='macro')
    
    model.cpu()
    
    return test_loss, test_f1, test_auc
