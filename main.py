# use model as ViViT model or slowfast model
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.dataloader import VideoDataset
from src.utils.sampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, List, Tuple
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pytorch_model_summary import summary

from src.utils.utility import video2tensor

# Module for ViViT
# Residual module
class Residule(nn.Module):
    def __init__(self, fn):
        super(Residule, self).__init__()
        self.fn = fn
    def forward(self, x : torch.Tensor, **kwargs)->torch.Tensor:
        return self.fn(x, **kwargs) + x

# PreNorm module
class PreNorm(nn.Module):
    def __init__(self, dim : int, fn : Union[nn.Module, None]):
        super(PreNorm, self).__init__()
        self.dim = dim
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x : torch.Tensor, **kwargs)->torch.Tensor:
        return self.fn(self.norm(x), **kwargs)

# FeedForward module
class FeedForward(nn.Module):
    def __init__(self, dim : int, hidden_dim : int, dropout : float = 0.5):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x : torch.Tensor)->torch.Tensor:
        return self.net(x)

# Attention
class Attention(nn.Module):
    def __init__(self, dim : int, n_heads : int = 8, d_head : int = 64, dropout : float = 0.5):
        super(Attention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = d_head
        self.dropout = dropout

        project_out = not (n_heads == 1 and d_head == dim)

        self.inner_dim = d_head * n_heads
        self.scale = d_head ** (-0.5)

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias = False) # q, k, v : (dim, inner_dim) matrix

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x : torch.Tensor)->torch.Tensor:
        b, n, _, h = *x.shape, self.n_heads

        qkv = self.to_qkv(x)
        qkv = torch.chunk(qkv, 3, -1)

        q,k,v = map(lambda t : rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # obtain attention score
        attn = torch.softmax(dots, dim = -1)

        # multply attention score * value
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        # rearrange as (b,n,h*d) => n_heads * d_head
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out

    
class Transformer(nn.Module):
    def __init__(self, dim : int, depth : int, n_heads : int, d_head : int, mlp_dim : int, dropout : float = 0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=dim, n_heads = n_heads, d_head = d_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim=dim, hidden_dim = mlp_dim, dropout = dropout))
            ]))

    def forward(self, x : torch.Tensor)->torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViViT(nn.Module):
    def __init__(
        self, 
        image_size : int, 
        patch_size : int, 
        n_frames : int, 
        dim : int = 192, 
        depth : int = 4, 
        n_heads : int = 3, 
        pool : str = 'cls', 
        in_channels : int = 3, 
        d_head :int = 64, 
        dropout : float = 0.,
        embedd_dropout : float = 0., 
        scale_dim :int = 4, 
        ):
        super(ViViT, self).__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls(cls token) or mean(mean pooling)'
        assert image_size % patch_size == 0, "Image dimension(height and width) must be divisible by the patch_size"

        self.image_size = image_size
        self.n_frames = n_frames
        self.in_channels = in_channels

        n_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_frames, n_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(
            dim, depth, n_heads, d_head, dim * scale_dim, dropout
        )

        self.temporal_token = nn.Parameter(torch.randn(1,1,dim))
        self.temporal_transformer = Transformer(
            dim, depth, n_heads, d_head, dim * scale_dim, dropout
        )

        self.dropout = nn.Dropout(embedd_dropout)
        self.pool = pool

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )

    def forward(self, x : torch.Tensor)->torch.Tensor:

        if x.size()[1] == self.in_channels:
            x = torch.permute(x, (0,2,1,3,4))
            
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape
        cls_space_token = repeat(self.space_token, '() n d -> b t n d', b = b, t = t)
        cls_temporal_token = repeat(self.temporal_token, '() n d -> b n d', b = b)

        x = torch.cat((cls_space_token, x), dim = 2)
        x += self.pos_embedding[:,:,:(n+1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:,0], '(b t) ... -> b t ...', b = b)

        x = torch.cat((cls_temporal_token, x), dim = 1)
        x = self.temporal_transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp(x)
        x = torch.sigmoid(x)

        return x

    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = True, print_summary : bool = True, show_parent_layers : bool = True)->None:
        sample = torch.zeros((1, self.n_frames, self.in_channels, self.image_size, self.image_size), device = device)
        return summary(self, sample, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers)

def compute_focal_loss(inputs:torch.Tensor, gamma:float):
    p = torch.exp(-inputs)
    loss = (1-p) ** gamma * inputs
    return loss.mean()

class FocalLossLDAM(torch.nn.Module):
    def __init__(self, weight : Optional[torch.Tensor] = None, gamma : float = 0.1):
        super(FocalLossLDAM, self).__init__()
        assert gamma >= 0, "gamma should be positive"
        self.gamma = gamma
        self.weight = weight
    
    def update_weight(self, weight : Optional[torch.Tensor] = None, gamma : float = 0.1):
        self.gamma = gamma
        self.weight = weight

    def forward(self, input : torch.Tensor, target : torch.Tensor)->torch.Tensor:
        # return compute_focal_loss(torch.nn.BCELoss()(input, target), self.gamma)
        return compute_focal_loss(F.cross_entropy(input, target, reduction = 'mean', weight = self.weight), self.gamma)

# Label-Distribution-Aware Margin loss
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list : Optional[List], max_m : float = 0.5, weight : Optional[torch.Tensor] = None, s : int = 30):
        super(LDAMLoss, self).__init__()
        assert s > 0, "s should be positive"
        self.s = s
        self.weight = weight
        
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list

    def update_m_list(self, cls_num_list : List, max_m : float, weight : torch.Tensor, s : int):
        self.s = s
        self.weight = weight
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list

    def forward(self, x : torch.Tensor, target : torch.Tensor)->torch.Tensor:

        idx = torch.zeros((x.size(0), 2), dtype = torch.uint8).to(x.device)
        idx.scatter_(1, target.data.view(-1,1), 1)
        idx_float = idx.type(torch.FloatTensor)

        batch_m = torch.matmul(self.m_list[None, :].to(x.device), idx_float.transpose(0,1).to(x.device))
        batch_m = batch_m.view((-1,1))
        x_m = x - batch_m

        output = torch.where(idx, x_m, x)
        return F.cross_entropy(self.s * output, target, weight = self.weight)

from typing import Optional, List
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report

def adjust_learning_rate(optimizer : torch.optim.Optimizer, epoch : int, args = None):
    epoch = epoch + 1

    if epoch <=5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def train_per_epoch(
    train_loader : torch.utils.data.DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    max_norm_grad : Optional[float] = None
    ):

    model.train()
    model.to(device)

    train_loss = 0
    train_acc = 0

    total_pred = np.array([])
    total_label = np.array([])

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        # for BCELoss
        # target = target.float().to(device)
        # output = model(data).squeeze(dim = 1)

        # For LDAMLoss
        target = target.to(device)
        output = model(data)

        loss = loss_fn(output, target)

        loss.backward()

        # use gradient clipping
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)

        optimizer.step()

        train_loss += loss.item()

        pred = (output > torch.FloatTensor([0.5]).to(device))
        train_acc += pred.eq(target.view_as(pred)).sum().item() / data.size(0) 
        
        total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
        total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))
        
    if scheduler:
        scheduler.step()

    train_loss /= (batch_idx + 1)
    train_acc /= (batch_idx + 1)

    train_f1 = f1_score(total_label, total_pred, average = "macro")

    return train_loss, train_acc, train_f1

def valid_per_epoch(
    valid_loader : torch.utils.data.DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
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

            # for BCELoss
            # target = target.float().to(device)
            # output = model(data).squeeze(dim = 1)

            # For LDAMLoss
            target = target.to(device)
            output = model(data)

            loss = loss_fn(output, target)
    
            valid_loss += loss.item()
            pred = (output >= torch.FloatTensor([0.5]).to(device))
            valid_acc += pred.eq(target.view_as(pred)).sum().item() / data.size(0) 

            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    valid_loss /= (batch_idx + 1)
    valid_acc /= (batch_idx + 1)

    valid_f1 = f1_score(total_label, total_pred, average = "macro")

    return valid_loss, valid_acc, valid_f1

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
    save_last_dir : str = "./weights/last.pt",
    max_norm_grad : Optional[float] = None,
    criteria : str = "f1_score"
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

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction = 'mean')

    for epoch in tqdm(range(num_epoch), desc = "training process"):

        train_loss, train_acc, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad
        )

        valid_loss, valid_acc, valid_f1 = valid_per_epoch(
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

        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train acc : {:.3f}, valid acc : {:.3f}, train f1 : {:.3f}, valid f1 : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_acc, valid_acc, train_f1, valid_f1
                ))

        # save the best parameters
        if save_best_only:
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

def train_LDAM_process(
    train_loader : torch.utils.data.DataLoader,
    valid_loader : torch.utils.data.DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : Union[LDAMLoss, FocalLossLDAM],
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : int = 1,
    save_best_only : bool = False,
    save_best_dir : str = "./weights/best.pt",
    save_last_dir : str = "./weights/last.pt",
    max_norm_grad : Optional[float] = None,
    criteria : str = "f1_score",
    cls_num_list : Optional[List] = None,
    ):

    train_loss_list = []
    valid_loss_list = []

    train_f1_list = []
    valid_f1_list = []

    best_f1 = 0
    best_epoch = 0
    best_loss = torch.inf

    idx = 0
    beta = 0

    for epoch in tqdm(range(num_epoch), desc = "training process"):
        idx = epoch // int(num_epoch / 2)
        betas = [0, 0.9999]
        beta = betas[idx]
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        
        # for LDAM
        # loss_fn.update_m_list(cls_num_list, 0.5, per_cls_weights, 30)

        # for FocalLoss
        loss_fn.update_weight(per_cls_weights, 0.5)

        train_loss, train_acc, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            None,
            loss_fn,
            device,
            max_norm_grad
        )

        valid_loss, valid_acc, valid_f1 = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device 
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train f1 : {:.3f}, valid f1 : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_f1, valid_f1
                ))

        # save the best parameters
        if save_best_only:
            if criteria == "f1_score" and best_f1 < valid_f1:
                best_f1 = valid_f1
                best_loss = valid_loss
                best_epoch  = epoch
                torch.save(model.state_dict(), save_best_dir)
            elif criteria == "loss" and best_loss > valid_loss:
                best_f1 = valid_f1
                best_loss = valid_loss
                best_epoch  = epoch
                torch.save(model.state_dict(), save_best_dir)

        # save the last parameters
        torch.save(model.state_dict(), save_last_dir)

    # print("\n============ Report ==============\n")
    print("training process finished, best loss : {:.3f} and best f1 : {:.3f}, best epoch : {}".format(
        best_loss, best_f1, best_epoch
    ))

    return  train_loss_list, train_f1_list,  valid_loss_list, valid_f1_list

def evaluate(
    test_loader : torch.utils.data.DataLoader, 
    model : torch.nn.Module,
    optimizer : Optional[torch.optim.Optimizer],
    loss_fn = None,
    device : Optional[str] = "cpu",
    save_dir : Optional[str] = None
):
    test_loss = 0
    test_acc = 0
    total_pred = np.array([])
    total_label = np.array([])

    if device is None:
        device = torch.device("cuda:0")

    model.to(device)
    model.eval()

    for idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            data = data.to(device)
            target = target.float().to(device)
            output = model(data).squeeze(dim = 1)

            loss = loss_fn(output, target)
    
            test_loss += loss.item()
            pred = (output > torch.FloatTensor([0.5]).to(device))
            test_acc += pred.eq(target.view_as(pred)).sum().item() / data.size(0) 

            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    test_loss /= (idx + 1)
    test_acc /= (idx + 1)
    
    test_f1 = f1_score(total_label, total_pred, average = "macro")
    conf_mat = confusion_matrix(total_label,  total_pred)

    plt.figure()
    sns.heatmap(
        conf_mat / np.sum(conf_mat, axis = 1)[:, None],
        annot = True,
        fmt = '.2f',
        cmap = 'Blues',
        xticklabels=["disruption", "normal"],
        yticklabels=["disruption", "normal"]
    )

    plt.savefig("./results/confusion_matrix.png")

    print("############### Classification Report ####################")
    print(classification_report(total_label, total_pred, labels = [0,1]))
    print("\n# test acc : {:.2f}, test f1 : {:.2f}, test loss : {:.3f}".format(test_acc, test_f1, test_loss))
    print(conf_mat)

    if save_dir:
        with open(save_dir, 'w') as f:
            f.write(classification_report(total_label, total_pred, labels = [0,1]))
            summary = "\n# test score : {:.2f}, test loss : {:.3f}, test f1 : {:.3f}".format(test_acc, test_loss, test_f1)
            f.write(summary)

    return test_loss, test_acc, test_f1

augmentation_args = {
    "bright_val" : 30,
    "bright_p" : 0.5,
    "contrast_min" : 1,
    "contrast_max" : 1.5,
    "contrast_p" : 0.5,
    "blur_k" : 5,
    "blur_p" : 0.5,
    "flip_p" : 0.5,
    "vertical_ratio" : 0.2,
    "vertical_p" : 0.5,
    "horizontal_ratio" : 0.2,
    "horizontal_p" : 0.5
}

image_size = 224
patch_size = 16

use_focal_loss = True
use_sampler = False

batch_size = 32
lr = 1e-3
clip_len = 21
num_epoch = 12
verbose = 1
gamma = 0.95
save_best_dir = "./weights/ViViT_clip_21_dist_0_best.pt"
save_last_dir = "./weights/ViViT_clip_21_dist_0_last.pt"
save_result_dir = "./results/train_valid_loss_acc_ViViT_clip_21_dist_0.png"
save_test_result = "./results/test_ViViT_clip_21_dist_0.txt"
dataset = "dur0.1_dis0"

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:" + str(0)
else:
    device = 'cpu'
    
if __name__ == "__main__":

    train_data = VideoDataset(dataset = dataset, split = "train", clip_len = clip_len, preprocess = False, augmentation = True, augmentation_args=augmentation_args)
    valid_data = VideoDataset(dataset = dataset, split = "val", clip_len = clip_len, preprocess = False, augmentation=True, augmentation_args=augmentation_args)
    test_data = VideoDataset(dataset = dataset, split = "test", clip_len = clip_len, preprocess = False, augmentation=False)

    if use_sampler:
        train_sampler = ImbalancedDatasetSampler(train_data)
        valid_sampler = ImbalancedDatasetSampler(valid_data)
        test_sampler = None

    else:
        train_sampler = None
        valid_sampler = None
        test_sampler = None
    
    train_loader = DataLoader(train_data, batch_size, sampler=train_sampler, num_workers = 8)
    valid_loader = DataLoader(valid_data, batch_size, sampler=valid_sampler, num_workers = 8)
    test_loader = DataLoader(test_data, batch_size, sampler=test_sampler, num_workers = 8)

    sample_data, sample_label = next(iter(train_loader))

    model = ViViT(
        image_size = image_size,
        patch_size = patch_size,
        n_frames = clip_len,
        dim = 64,
        depth = 4,
        n_heads = 4,
        pool = "cls",
        in_channels = 3,
        d_head = 64,
        dropout = 0.25,
        embedd_dropout=0.25,
        scale_dim = 4
    )

    model.cpu()
    model.to(device)

    model.summary(device, show_input = True, show_hierarchical=False, print_summary=True, show_parent_layers=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=gamma)

    # scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer,
    #         step_size = 2,
    #         gamma = gamma
    # )
    
    # if use_focal_loss:
    #     loss_fn = FocalLossLDAM(weight = None, gamma = 0.5)
    # else: 
    #     loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")

    # # training process
    # train_loss,  train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train(
    #     train_loader,
    #     valid_loader,
    #     model,
    #     optimizer,
    #     scheduler,
    #     loss_fn,
    #     device,
    #     num_epoch,
    #     verbose,
    #     save_best_only = False,
    #     save_best_dir = save_best_dir,
    #     save_last_dir = save_last_dir,
    #     max_norm_grad = 5.0,
    #     criteria = "f1_score"
    # )

    train_data.get_img_num_per_cls()
    cls_num_list = train_data.get_cls_num_list()

    print("cls_num_list : ", cls_num_list)

    # loss_fn = LDAMLoss(cls_num_list, max_m = 0.5, s = 30, weight = None)

    loss_fn = FocalLossLDAM(weight = None, gamma = 0.5)

    train_loss, train_f1, valid_loss, valid_f1 = train_LDAM_process(
        train_loader,
        valid_loader,
        model,
        optimizer,
        loss_fn,
        device,
        num_epoch,
        verbose,
        True,
        save_best_dir,
        save_last_dir,
        max_norm_grad = 5.0,
        criteria = "f1_score",
        cls_num_list = cls_num_list
    )

    model.load_state_dict(torch.load(save_last_dir))

    # evaluation process
    test_loss, test_acc, test_f1 = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
        save_test_result
    )

    model.cpu()