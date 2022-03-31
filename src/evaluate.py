import numpy as np
import torch
from typing import Optional
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.dataloader import VideoDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true)**2)

def Corr(pred, true):
    sig_p = np.std(pred, axis = 0)
    sig_g = np.std(true, axis = 0)
    m_p = pred.mean(0)
    m_g = true.mean(0)
    int = (sig_g != 0)


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
            target = target.to(device)
            output = model.forward(data)

            loss = loss_fn(output, target)
    
            test_loss += loss.item()
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            test_acc += pred.eq(target.view_as(pred)).sum().item() / data.size(0) 

            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    test_loss /= (idx + 1)
    test_acc /= (idx + 1)

    conf_mat = confusion_matrix(total_label,  total_pred)

    plt.figure()
    sns.heatmap(
        conf_mat / np.sum(conf_mat, axis = 1),
        annot = True,
        fmt = '.2f',
        cmap = 'Blues',
        xticklabels=[0,1],
        yticklabels=[0,1]
    )

    plt.savefig("./results/confusion_matrix.png")

    print("############### Classification Report ####################")
    print(classification_report(total_label, total_pred, labels = [0,1]))
    print("\n# total test score : {:.2f} and test loss : {:.3f}".format(test_acc, test_loss))
    print(conf_mat)

    if save_dir:
        with open(save_dir, 'w') as f:
            f.write(classification_report(total_label, total_pred, labels = [0,1]))
            summary = "\n# total test score : {:.2f} and test loss : {:.3f}".format(test_acc, test_loss)
            f.write(summary)

    return test_loss, test_acc