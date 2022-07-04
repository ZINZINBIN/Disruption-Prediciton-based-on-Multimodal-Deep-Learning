from typing import Optional, List
import warnings
from src.loss import LDAMLoss
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.dataloader import VideoDataset
from src.utils.mixup import mixup_criterion, video_mixup_data
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

def train_per_epoch(
    train_loader : torch.utils.data.DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    use_video_mixup : bool = False,
    ):

    model.train()
    model.to(device)

    train_loss = 0
    train_acc = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        target_ = None
        lam = None

        if use_video_mixup:
            data, target, target_, lam = video_mixup_data(data, target, device, "spatial-temporal", alpha = 1.0)
            data, target, target_ = map(torch.autograd.Variable, (data, target, target_))

        output = model(data)

        if use_video_mixup:
            loss = mixup_criterion(loss_fn, output, target, target_, lam)
        else:
            loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if use_video_mixup:
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            train_acc += (lam * pred.eq(target.view_as(pred)).sum().item() + (1-lam) * pred.eq(target_.view_as(pred)).sum().item()) / data.size(0) 
            
        else:
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            train_acc += pred.eq(target.view_as(pred)).sum().item() / data.size(0) 

    if scheduler:
        scheduler.step()

    train_loss /= (batch_idx + 1)
    train_acc /= (batch_idx + 1)

    return train_loss, train_acc

def valid_per_epoch(
    valid_loader : torch.utils.data.DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    use_acc_per_class : Optional[bool] = False
    ):

    model.eval()
    model.to(device)
    valid_loss = 0
    valid_acc = 0

    total_pred = np.array([])
    total_label = np.array([])

    for batch_idx, (data, target) in enumerate(valid_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            loss = loss_fn(output, target)
    
            valid_loss += loss.item()
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            valid_acc += pred.eq(target.view_as(pred)).sum().item() / data.size(0) 

            if use_acc_per_class:
                total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
                total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    valid_loss /= (batch_idx + 1)
    valid_acc /= (batch_idx + 1)

    if use_acc_per_class:
        conf_mat = confusion_matrix(total_label,  total_pred)
        return valid_loss, valid_acc, conf_mat
    else:
        return valid_loss, valid_acc


def train(
    train_loader : torch.utils.data.DataLoader, 
    valid_loader : torch.utils.data.DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn = None,
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_only : bool = False,
    save_best_dir : str = "./weights/best.pt",
    use_video_mixup_algorithm : bool = False
):

    train_loss_list = []
    valid_loss_list = []
    
    train_acc_list = []
    valid_acc_list = []

    best_acc = 0
    best_epoch = 0
    best_loss = torch.inf

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction = 'mean')


    for epoch in tqdm(range(num_epoch), desc = "training process"):

        train_loss, train_acc = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            use_video_mixup_algorithm 
        )

        valid_loss, valid_acc = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device 
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train acc : {:.3f}, valid acc : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_acc, valid_acc
                ))

        if save_best_only:
            if best_acc < valid_acc:
                best_acc = valid_acc
                best_loss = valid_loss
                best_epoch  = epoch
                torch.save(model.state_dict(), save_best_dir)

    # print("\n============ Report ==============\n")
    print("training process finished, best loss : {:.3f} and best acc : {:.3f}, best epoch : {}".format(
        best_loss, best_acc, best_epoch
    ))

    return  train_loss_list, train_acc_list,  valid_loss_list,  valid_acc_list


def train_(
    train_loader : torch.utils.data.DataLoader, 
    valid_loader : torch.utils.data.DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : Optional[torch.nn.Module],
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best : Optional[str] = "./weights/best.pt",
    use_video_mixup : bool = False,
    use_acc_per_class : bool = False,
    train_rule : Optional[str] = None,
    cls_num_list : Optional[List] = None
):

    train_loss_list = []
    valid_loss_list = []
    
    train_acc_list = []
    valid_acc_list = []

    best_acc = 0
    best_epoch = 0
    best_loss = torch.inf

    if loss_fn is None and train_rule is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction = 'mean')
    
    if cls_num_list is None and train_rule == "DRW":
        warnings.warn("cls_num_list should be necessay for DRW algorithm : change train rule as None")
        loss_fn = torch.nn.CrossEntropyLoss(reduction = 'mean')
        train_rule = None

    for epoch in tqdm(range(num_epoch), desc = "training process"):

        # train rule : DRW should be updated after some epochs passed
        if train_rule == 'DRW':
            idx = epoch // int(num_epoch / 2)
            betas = [0, 0.9999]
            beta = betas[idx]
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
            loss_fn = LDAMLoss(cls_num_list, max_m = 0.5, weight = per_cls_weights, s = 30)

        # train process for 1 epoch
        train_loss, train_acc = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            use_video_mixup
        )

        # validation process for 1 epoch
        if use_acc_per_class:
            valid_loss, valid_acc, conf_mat = valid_per_epoch(
                valid_loader, 
                model,
                optimizer,
                loss_fn,
                device,
                use_acc_per_class
            )

        else:
            valid_loss, valid_acc = valid_per_epoch(
                valid_loader, 
                model,
                optimizer,
                loss_fn,
                device,
                False
            )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train acc : {:.3f}, valid acc : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_acc, valid_acc
                ))

        if save_best:
            if best_acc < valid_acc:
                best_acc = valid_acc
                best_loss = valid_loss
                best_epoch  = epoch
                torch.save(model.state_dict(), save_best)

    # print("\n============ Report ==============\n")
    print("training process finished, best loss : {:.3f} and best acc : {:.3f}, best epoch : {}".format(
        best_loss, best_acc, best_epoch
    ))

    return  train_loss_list, train_acc_list,  valid_loss_list,  valid_acc_list