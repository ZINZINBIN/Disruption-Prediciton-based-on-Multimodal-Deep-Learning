from sched import scheduler
from typing import Optional
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.dataloader import VideoDataset
from torch.utils.data import DataLoader
from src.model import R2Plus1DClassifier
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from src.evaluate import evaluate

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:0"
else:
    device = 'cpu'

if __name__ == "__main__":

    batch_size = 8
    test_data_dist10 = VideoDataset(dataset = "fast_model_dataset", split = "test", clip_len = 8, preprocess = False)
    test_loader_dist10 = DataLoader(test_data_dist10, batch_size = batch_size, shuffle = True, num_workers = 4)

    model = R2Plus1DClassifier(
        input_size  = (3, 8, 112, 112),
        num_classes = 2, 
        layer_sizes = [2,2,2,2], 
        pretrained = False, 
        alpha = 0.01
    )

    model.to(device)
    model.load_state_dict(torch.load("./weights/best.pt", map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")

    test_loss, test_acc = evaluate(
        test_loader_dist10,
        model,
        optimizer,
        loss_fn,
        device
    )