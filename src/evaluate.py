import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true)**2)

def evaluate(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : Optional[torch.optim.Optimizer],
    loss_fn : Optional[torch.nn.Module]= None,
    device : Optional[str] = "cpu",
    save_conf : Optional[str] = "./results/confusion_matrix.png",
    save_txt : Optional[str] = None,
    threshold : float = 0.5,
    ):

    test_loss = 0
    test_acc = 0
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
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model.forward(data)

            loss = loss_fn(output, target)
    
            test_loss += loss.item()
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            pred = (pred > torch.FloatTensor([threshold]).to(device))
            test_acc += pred.eq(target.view_as(pred)).sum().item()

            total_size += data.size(0)
            
            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    test_loss /= (idx + 1)
    test_acc /= total_size
    test_f1 = f1_score(total_label, total_pred, average = "macro")
    
    conf_mat = confusion_matrix(total_label, total_pred)

    if save_conf is None:
        save_conf = "./results/confusion_matrix.png"

    plt.figure()
    s = sns.heatmap(
        conf_mat, # conf_mat / np.sum(conf_mat),
        annot = True,
        fmt ='04d' ,# fmt = '.2f',
        cmap = 'Blues',
        xticklabels=["disruption","normal"],
        yticklabels=["disruption","normal"],
    )

    s.set_xlabel("Prediction")
    s.set_ylabel("Actual")

    plt.savefig(save_conf)

    print("############### Classification Report ####################")
    print(classification_report(total_label, total_pred, labels = [0,1]))
    print("\n# test acc : {:.2f}, test f1 : {:.2f}, test loss : {:.3f}".format(test_acc, test_f1, test_loss))

    if save_txt:
        with open(save_txt, 'w') as f:
            f.write(classification_report(total_label, total_pred, labels = [0,1]))
            summary = "\n# test score : {:.2f}, test loss : {:.3f}, test f1 : {:.3f}".format(test_acc, test_loss, test_f1)
            f.write(summary)

    return test_loss, test_acc, test_f1