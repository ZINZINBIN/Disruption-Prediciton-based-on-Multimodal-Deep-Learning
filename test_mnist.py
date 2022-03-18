import argparse
import torch
from src.utils_example import *
from src.layer import *
from src.model import MNIST_Net
from tqdm import tqdm

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

model =  MNIST_Net(
    input_shape  = (1, 28, 28),
    conv_channels  = [1, 8, 16], 
    conv_kernels = [8, 4],
    pool_strides =  [2, 2],
    pool_kernels = [2, 2],
    theta_dim = 64
)

model.to(device)

# loss, optimizer and scheudler
optimizer = torch.optim.AdamW(model.parameters(), lr =  1e-3)
loss_fn = torch.nn.NLLLoss(reduction = 'mean')

# train and evaluate
def train_per_epoch(model, train_loader, loss_fn, optimizer, device="cpu"):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = loss_fn(output,target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.view_as(pred)).sum().item() / data.size(0)

    train_loss /= (batch_idx + 1)
    train_acc /= (batch_idx + 1)

    return train_loss, train_acc

def train(model, train_loader, loss_fn, optimizer, num_epoch = 64, device="cpu", verbose = 8):
    for epoch in tqdm(range(num_epoch)):
        train_loss, train_acc = train_per_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        if epoch % verbose == 0:
            print("train epoch : {}, train loss : {:.3f}, train acc : {:.3f}".format(epoch,  train_loss, train_acc))

def evaluate(model, test_loader, device = 'cpu'):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_acc = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output,target).item()
            pred = output.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= (batch_idx + 1)
        test_acc /= (batch_idx + 1)

        print("test set : loss = {:.3f} and acc : {:.3f}".format(test_loss, test_acc))

if __name__ =='__main__':
    train_loader = mnist_train_loader
    test_loader = mnist_test_loader

    sample_data, sample_target = next(iter(train_loader))

    print("batch data size : ", sample_data.size())
    print("batch target size :", sample_target.size())

    train(model,train_loader,loss_fn,optimizer,64,device,8)
    evaluate(model,test_loader,device)