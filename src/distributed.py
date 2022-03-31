from torch import triplet_margin_loss
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional
import numpy as np
import random

def set_random_seeds(random_seed : int = 42):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_distributed_loader(train_dataset : Dataset, valid_dataset : Dataset, num_replicas : int, rank : int, num_workers : int, batch_size : int = 32):
    train_sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank = rank, shuffle = True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=num_replicas, rank = rank, shuffle = True)

    train_loader = DataLoader(train_dataset, batch_size,  sampler = train_sampler, num_workers = num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size,  sampler = valid_sampler, num_workers = num_workers, pin_memory=True)

    return train_loader, valid_loader

def train_epoch_per_procs(
    rank : int, 
    world_size : int, 
    batch_size : Optional[int],
    model : torch.nn.Module,
    train_dataset : Dataset,
    valid_dataset : Dataset,
    random_seed : int = 42,
    resume : bool = True,
    loss_fn = None,
    model_filepath : str = "./weights/distributed.pt"
    ):

    device = torch.device("cuda:{}".format(rank))
    set_random_seeds(random_seed)

    model.to(device)
    ddp_model = DDP(model, device_ids = [rank], output_device=rank)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction = "mean")

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr = 2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 8, T_mult = 2)

    if not os.path.isfile(model_filepath) and dist.get_rank() == 0:
        torch.save(model.state_dict(), model_filepath)

    dist.barrier()

    if resume == True:
        map_location = {"cuda:0":"cuda:{}".format(rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location), strict = False)

    train_loader, valid_loader = get_distributed_loader(train_dataset, valid_dataset, num_replicas=world_size, rank = rank, num_workers = 4, batch_size = batch_size)

    # train process
    model.train()
    train_loss = 0
    train_acc = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
        train_acc += pred.eq(target.view_as(pred)).sum().item() / data.size(0) 

    if scheduler:
        scheduler.step()

    train_loss /= (batch_idx + 1)
    train_acc /= (batch_idx + 1)

    # valid process
    model.eval()
    valid_loss = 0
    valid_acc = 0

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

    return train_loss, train_acc, valid_loss, valid_acc

def train_per_proc(
    rank : int, 
    world_size : int, 
    batch_size : Optional[int],
    model : torch.nn.Module,
    train_dataset : Dataset,
    valid_dataset : Dataset,
    random_seed : int = 42,
    resume : bool = True,
    loss_fn = None,
    model_filepath : str = "./weights/distributed.pt",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_only : bool = False,
    save_best_dir : str = "./weights/best.pt"
):
    dist.init_process_group("nccl", rank = rank, world_size = world_size)

    train_loss_list = []
    valid_loss_list = []
    
    train_acc_list = []
    valid_acc_list = []

    best_acc = 0
    best_epoch = 0
    best_loss = torch.inf

    for epoch in range(num_epoch):

        train_loss, train_acc, valid_loss, valid_acc = train_epoch_per_procs(
            rank,
            world_size,
            batch_size,
            model,
            train_dataset,
            valid_dataset,
            random_seed = 42,
            resume = True,
            loss_fn = loss_fn,
            model_filepath = model_filepath
        )

        dist.barrier()

        if dist.get_rank() == 0:
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)

            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

            if verbose:
                if epoch % verbose == 0:
                    print("rank : {}, epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train acc : {:.3f}, valid acc : {:.3f}".format(
                        rank, epoch+1, train_loss, valid_loss, train_acc, valid_acc
                    ))

            if save_best_only:
                if best_acc < valid_acc:
                    best_acc = valid_acc
                    best_loss = valid_loss
                    best_epoch  = epoch
                    torch.save(model.state_dict(), save_best_dir)
            else:
                torch.save(model.state_dict(), model_filepath)

    if dist.get_rank() == 0:
        # print("\n============ Report ==============\n")
        print("training process finished, best loss : {:.3f} and best acc : {:.3f}, best epoch : {}".format(
            best_loss, best_acc, best_epoch
        ))

    return  train_loss_list, train_acc_list, valid_loss_list,  valid_acc_list


def train_distributed(
    world_size : int, 
    batch_size : Optional[int],
    model : torch.nn.Module,
    train_dataset : Dataset,
    valid_dataset : Dataset,
    random_seed : int = 42,
    resume : bool = True,
    loss_fn = None,
    model_filepath : str = "./weights/distributed.pt",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_only : bool = False,
    save_best_dir : str = "./weights/distributed_best.pt"
    ):

    if world_size > torch.cuda.device_count():
        world_size = torch.cuda.device_count()

    mp.spawn(
        train_per_proc,
        args = (world_size,batch_size, model, train_dataset,valid_dataset, random_seed, resume, loss_fn, model_filepath, num_epoch, verbose, save_best_only, save_best_dir),
        nprocs = world_size,
        join = True
    )


def example(rank, world_size):
    dist.init_process_group("gloo", rank = rank, world_size=world_size)
    model = nn.Linear(10,10).to(rank)
    ddp_model = DDP(model, device_ids = [rank])

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr = 1e-3)

    outputs = ddp_model(torch.randn(20,10).to(rank))
    labels = torch.randn(20,10).to(rank)
    
    optimizer.zero_grad()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    print("rank : {} process".format(rank))

def main():
    world_size = 2
    mp.spawn(
        example,
        args = (world_size, ),
        nprocs = world_size,
        join = True
    )

import os

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()