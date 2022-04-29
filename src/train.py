from typing import Optional
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
    loss_fn = None,
    device : str = "cpu",
    use_video_mixup_algorithm : bool = False,
    use_acc_per_class : bool = False
    ):

    model.train()
    model.to(device)
    train_loss = 0
    train_acc = 0

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction = 'mean')

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        target_ = None
        lam = None

        if use_video_mixup_algorithm:
            data, target, target_, lam = video_mixup_data(data, target, device, "spatial-temporal", alpha = 1.0)
            data, target, target_ = map(torch.autograd.Variable, (data, target, target_))

        output = model(data)

        if use_video_mixup_algorithm:
            loss = mixup_criterion(loss_fn, output, target, target_, lam)
        else:
            loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if use_video_mixup_algorithm:
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
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn = None,
    device : str = "cpu"
):
    model.eval()
    model.to(device)
    valid_loss = 0
    valid_acc = 0

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")

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

    valid_loss /= (batch_idx + 1)
    valid_acc /= (batch_idx + 1)

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
            scheduler,
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