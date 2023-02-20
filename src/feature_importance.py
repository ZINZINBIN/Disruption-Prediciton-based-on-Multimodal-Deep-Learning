''' 
Codes for Permutation importance and Perturbation importance
Permutation feature importance algorithm
    Input : trained model, feature matrix, target vector, error measure
    1. Estimate the original model error e_orig = L(y,f(x))
    2. For each feature j, do the processes below
        - Generate feature matrix X_perm by permuting feature j in the data X. 
        - Estimate error e_perm = L(y, f(X_perm)) based on the predictions of the permuted data
        - Calculate permutation feature matrix as quotient FI_j = e_perm / e_orig
    3. Sort features by descending FI
Reference
- https://towardsdatascience.com/neural-feature-importance-1c1868a4bf53
- https://github.com/pytorch/captum
- https://moondol-ai.tistory.com/401
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Literal, List
from sklearn.metrics import f1_score

def compute_loss(
    dataloader : DataLoader, 
    model : nn.Module,
    loss_fn : nn.Module,
    device : str = "cpu",
    model_type : Literal["single","multi","multi-GB"] = "single"
    ):

    model.eval()
    model.to(device)

    total_loss = 0
    total_f1 = 0
    total_pred = np.array([])
    total_label = np.array([])
    total_size = 0

    for batch_idx, (data, target) in enumerate(dataloader):
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
    
            total_loss += loss.item()
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            total_size += pred.size(0)

            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    total_f1 = f1_score(total_label, total_pred, average = "macro")

    return total_loss, total_f1


# permutation importance
def compute_permute_feature_importance(
    model : nn.Module,
    dataloader : DataLoader,
    features : List,
    loss_fn : nn.Module,
    device : str,
    model_type : Literal["single","multi","multi-GB"],
    criteria : Literal['loss','score'],
    save_dir : str
    ):
    
    # convert get_shot_num variable true
    dataloader.dataset.get_shot_num = False

    n_features = len(features)
    data_orig = dataloader.dataset.ts_data.copy()
    
    results = []
    
    loss_orig, score_orig = compute_loss(dataloader, model, loss_fn, device, model_type)
    
    for k in tqdm(range(n_features), desc = "processing for feature importance"):
        
        # permutate the k-th features
        np.random.shuffle(dataloader.dataset.ts_data[features[k]].values)
        
        # compute the loss
        loss, score = compute_loss(dataloader, model, loss_fn, device, model_type)
        
        # return the order of the features
        dataloader.dataset.ts_data = data_orig
        
        if criteria == 'loss':
            fi = abs(abs(loss - loss_orig) / loss_orig)
        else:
            fi = abs(score - score_orig) / score_orig
        
        # update result
        results.append({"feature":features[k], "loss":loss, "score":score, "feature_importance":fi})
        
    df = pd.DataFrame(results)
    df = df.sort_values('feature_importance')
    
    plt.figure(figsize = (8,8))
    plt.barh(np.arange(n_features), df.feature_importance)
    plt.yticks(np.arange(n_features), df.feature.values)
    plt.title('0D data - feature importance')
    plt.ylim((-1,n_features + 1))
    plt.xlim([0, 5.0])
    plt.xlabel('Permutation feature importance')
    plt.ylabel('Feature', size = 14)
    plt.savefig(save_dir)