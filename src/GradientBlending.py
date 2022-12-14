# Gradient Blending for multimodal fusion
# To avoid overfitting from multimodal training, we use gradient blending method
# see reference : https://arxiv.org/abs/1905.12681
# online G-Blend and offline G-Blend
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from typing import Optional, Literal, Dict, Union
from src.models.MultiModal import TFN_GB, MultiModalModel_GB
from src.train import train_per_epoch, valid_per_epoch

# Gradient Bleding with weighted loss sum
class GradientBlending(nn.Module):
    def __init__(
        self, 
        loss_vis : nn.Module,
        loss_ts : nn.Module,
        loss_vis_ts : nn.Module,
        vis_weight : float = 0.0, 
        ts_weight : float = 0.0, 
        vis_ts_weight : float = 1.0, 
        loss_scale : float = 1.0
        ):
        super(GradientBlending, self).__init__()
        self.loss_vis = loss_vis
        self.loss_ts = loss_ts
        self.loss_vis_ts = loss_vis_ts
        self.vis_weight = vis_weight
        self.ts_weight = ts_weight
        self.vis_ts_weight = vis_ts_weight
        self.loss_scale = loss_scale
        
    def update_weights(self, ws : Dict):
        self.vis_weight = ws['video']
        self.ts_weight = ws['0D']
        self.vis_ts_weight = ws['multi']

    def forward(self, vis_ts_out : torch.Tensor, vis_out : torch.Tensor, ts_out : torch.Tensor, target : torch.Tensor):
        loss_vis = self.loss_vis(vis_out, target) * self.loss_scale
        loss_ts = self.loss_ts(ts_out, target) * self.loss_scale
        loss_vis_ts = self.loss_vis_ts(vis_ts_out, target) * self.loss_scale
        loss = loss_vis * self.vis_weight + loss_ts * self.ts_weight + loss_vis_ts * self.vis_ts_weight
        return loss
    
def GB_estimate(
    n_epochs : int, 
    train_loader : DataLoader, 
    valid_loader : DataLoader, 
    multi_save_dir : str,
    multi_model : nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : nn.Module,
    device : str = "cpu",
    max_norm_grad : Optional[float] = None,
    ):
    
    train_loss_list = []
    valid_loss_list = []
    w_dict = {}
    w_list = []
    
    tasks = ["video","0D","multi"]
    for task in tasks:
        
        multi_model.load_state_dict(torch.load(multi_save_dir))
        multi_model.update_use_stream(task)
        
        for epoch in range(n_epochs):
            train_loss, _, _ = train_per_epoch(
                train_loader, 
                multi_model,
                optimizer,
                scheduler,
                loss_fn,
                device,
                max_norm_grad,
                "multi"
            )

            # validation process
            valid_loss, _, _ = valid_per_epoch(
                valid_loader, 
                multi_model,
                optimizer,
                loss_fn,
                device,
                "multi"
            )

            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
        
        Oi = valid_loss_list[0] - train_loss_list[0]
        Of = valid_loss_list[-1] - train_loss_list[-1]
        G = valid_loss_list[-1] - valid_loss_list[0]
        
        w = G / (Of-Oi) ** 2
        w_list.append(w)
    
    keys = tasks
    w_list = np.array(w_list) / np.sum(w_list)
    
    for key, w in zip(keys, w_list):
        w_dict[key] = w
    
    return w_dict

def evaluate_GB(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : Optional[torch.optim.Optimizer],
    device : Optional[str] = "cpu",
    threshold : float = 0.5,
    ):
    
    total_pred_vis = np.array([])
    total_pred_0D = np.array([])
    total_pred = np.array([])
    total_label = np.array([])

    if device is None:
        device = torch.device("cuda:0")

    model.to(device)
    model.eval()

    total_size = 0

    for idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            data_video = data['video'].to(device)
            data_0D = data['0D'].to(device)
            output, output_vis, output_ts = model(data_video, data_0D)
            target = target.to(device)
            
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            pred_vis = torch.nn.functional.softmax(output_vis, dim = 1).max(1, keepdim = True)[1]
            pred_0D = torch.nn.functional.softmax(output_ts, dim = 1).max(1, keepdim = True)[1]
            
            pred = (pred > torch.FloatTensor([threshold]).to(device))
            pred_vis = (pred_vis > torch.FloatTensor([threshold]).to(device))
            pred_0D = (pred_0D > torch.FloatTensor([threshold]).to(device))
            total_size += pred.size(0)
            
            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_pred_vis = np.concatenate((total_pred_vis, pred_vis.cpu().numpy().reshape(-1,)))
            total_pred_0D = np.concatenate((total_pred_0D, pred_0D.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    test_f1_fusion = f1_score(total_label, total_pred, average = "macro")
    test_f1_vis = f1_score(total_label, total_pred_vis, average = "macro")
    test_f1_0D = f1_score(total_label, total_pred_0D, average = "macro")
    
    return test_f1_fusion, test_f1_vis, test_f1_0D

def train_GB(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : Union[TFN_GB, MultiModalModel_GB],
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_GB : GradientBlending,
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_dir : str = "./weights/best.pt",
    save_last_dir : str = "./weights/last.pt",
    max_norm_grad : Optional[float] = None,
    criteria : Literal["f1_score", "acc", "loss"] = "f1_score",
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

    for epoch in tqdm(range(num_epoch), desc = "(multi-modal) training process with Gradient Blending"):
        
        # training process
        train_loss, train_acc, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_GB,
            device,
            max_norm_grad,
            "multi-GB"
        )

        # validation process
        valid_loss, valid_acc, valid_f1 = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_GB,
            device,
            "multi-GB"
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)
        
        # evaluate process : monitoring
        train_f1_fusion, train_f1_vis, train_f1_0D = evaluate_GB(
            train_loader,
            model,
            optimizer,
            device,
            0.5
        )
        
        valid_f1_fusion, valid_f1_vis, valid_f1_0D = evaluate_GB(
            valid_loader,
            model,
            optimizer,
            device,
            0.5
        )

        if verbose:
            if epoch % verbose == 0:
                print("# epoch: {}, train loss: {:.3f}, valid loss: {:.3f}".format(epoch+1, train_loss, valid_loss))
                print("# train, fusion: {:.3f}, video: {:.3f}, 0D : {:.3f}".format(train_f1_fusion, train_f1_vis, train_f1_0D))
                print("# valid, fusion: {:.3f}, video: {:.3f}, 0D : {:.3f}".format(valid_f1_fusion, valid_f1_vis, valid_f1_0D))
        
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
    
    print("(Report) training process finished, best loss : {:.3f} and best acc : {:.3f}, best f1 : {:.3f}, best epoch : {}".format(
        best_loss, best_acc, best_f1, best_epoch
    ))
    
    return  train_loss_list, train_acc_list, train_f1_list,  valid_loss_list,  valid_acc_list, valid_f1_list

def train_GB_dynamic(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : Union[TFN_GB, MultiModalModel_GB],
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_GB : GradientBlending,
    loss_unimodal : nn.Module,
    device : str = "cpu",
    num_epoch : int = 64,
    epoch_per_GB_estimate : int = 16,
    num_epoch_GB_estimate : int = 4,
    verbose : Optional[int] = 8,
    save_best_dir : str = "./weights/best.pt",
    save_last_dir : str = "./weights/last.pt",
    max_norm_grad : Optional[float] = None,
    criteria : Literal["f1_score", "acc", "loss"] = "f1_score",
    ):
    
    model_type = "multi-GB"

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

    for epoch in tqdm(range(num_epoch), desc = "(multi-modal) training process with Gradient Blending"):
        
        model.update_use_stream("multi-GB")
        
        # training process
        train_loss, train_acc, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_GB,
            device,
            max_norm_grad,
            model_type
        )

        # validation process
        valid_loss, valid_acc, valid_f1 = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_GB,
            device,
            model_type
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train acc : {:.3f}, valid acc : {:.3f}, train f1 : {:.3f}, valid f1 : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_acc, valid_acc, train_f1, valid_f1
                ))
        
        if epoch % epoch_per_GB_estimate:
            ws = GB_estimate(num_epoch_GB_estimate, train_loader, valid_loader, save_last_dir, model, optimizer, scheduler, loss_unimodal, device, max_norm_grad)
            loss_GB.update_weights(ws)

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

    model.update_use_stream("multi-GB")
    
    print("(Report) training process finished, best loss : {:.3f} and best acc : {:.3f}, best f1 : {:.3f}, best epoch : {}".format(
        best_loss, best_acc, best_f1, best_epoch
    ))
    
    return  train_loss_list, train_acc_list, train_f1_list,  valid_loss_list,  valid_acc_list, valid_f1_list





