import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

def visualize_2D_latent_space(model : nn.Module, dataloader : DataLoader, device : str = 'cpu', save_dir : str = './results/latent_2d_space.png'):
    model.to(device)
    model.eval()
    
    total_label = np.array([])
    total_latent = []
        
    for idx, (data, target) in enumerate(dataloader):
        with torch.no_grad():
            data = data.to(device)
            latent = model.encode(data)
            batch = data.size()[0]
            
            total_latent.append(latent.detach().cpu().numpy().reshape(batch,-1))
            total_label = np.concatenate((total_label, target.detach().cpu().numpy().reshape(-1,)), axis = 0)
    
    total_latent = np.concatenate(total_latent, axis = 0)
    total_label = total_label.astype(int)
    
    color = np.array(['#1f77b4', '#ff7f0e'])
    label  = np.array(['disruption','normal'])
    
    pca = PCA(n_components=2, random_state=42)
    total_latent = pca.fit_transform(total_latent)
    
    dis_idx = np.where(total_label == 0)
    normal_idx = np.where(total_label == 1)
    
    plt.figure(figsize = (8,6))
    plt.scatter(total_latent[dis_idx,0], total_latent[dis_idx,1], c = color[0], label = label[0])
    plt.scatter(total_latent[normal_idx,0], total_latent[normal_idx,1], c = color[1], label = label[1])
    plt.xlabel('z-0')
    plt.ylabel('z-1')
    plt.legend()
    plt.savefig(save_dir)
    
def visualize_3D_latent_space(model : nn.Module, dataloader : DataLoader, device : str = 'cpu', save_dir : str = './results/latent_2d_space.png'):
    model.to(device)
    model.eval()
    
    total_label = np.array([])
    total_latent = []
        
    for idx, (data, target) in enumerate(dataloader):
        with torch.no_grad():
            data = data.to(device)
            latent = model.encode(data)
            batch = data.size()[0]
            
            total_latent.append(latent.detach().cpu().numpy().reshape(batch,-1))
            total_label = np.concatenate((total_label, target.detach().cpu().numpy().reshape(-1,)), axis = 0)
    
    total_latent = np.concatenate(total_latent, axis = 0)
    total_label = total_label.astype(int)
    
    color = np.array(['#1f77b4', '#ff7f0e'])
    label  = np.array(['disruption','normal'])
    
    pca = PCA(n_components=3, random_state=42)
    total_latent = pca.fit_transform(total_latent)
    
    dis_idx = np.where(total_label == 0)
    normal_idx = np.where(total_label == 1)
    
    fig = plt.figure(figsize = (8,6))
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(total_latent[dis_idx,0], total_latent[dis_idx,1], total_latent[dis_idx,2], c = color[0], label = label[0])
    ax.scatter(total_latent[normal_idx,0], total_latent[normal_idx,1], total_latent[normal_idx,2], c = color[1], label = label[1])
    ax.set_xlabel('z-0')
    ax.set_ylabel('z-1')
    ax.set_zlabel('z-2')
    ax.legend()
    plt.savefig(save_dir)