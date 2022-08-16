import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from src.CustomDataset import CustomDataset
from src.models.resnet import Bottleneck3D
from src.models.slowfast import SlowFastClassifier, SlowFastEncoder, SlowFast
from src.utils.sampler import ImbalancedDatasetSampler
from src.train import train, train_LDAM_process
from src.evaluate import evaluate
from src.loss import FocalLoss, LDAMLoss
from src.utils.utility import show_data_composition

parser = argparse.ArgumentParser(description="training SlowFast Disruption Classifier")

parser.add_argument("--batch_size", type = int, default = 16)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--gpu_num", type = int, default = 1)

parser.add_argument("--num_workers", type = int, default = 8)
parser.add_argument("--pin_memory", type = bool, default = False)

parser.add_argument("--alpha", type = int, default = 4)
parser.add_argument("--seq_len", type = int, default = 20)
parser.add_argument("--hidden", type = int, default = 64)
parser.add_argument("--image_size", type = int, default = 224)

parser.add_argument("--use_sampler", type = bool, default = True)
parser.add_argument("--num_epoch", type = int, default = 8)
parser.add_argument("--verbose", type = int, default = 1)

parser.add_argument("--save_best_dir", type = str, default = "./weights/slowfast_clip_21_dist_0_best.pt")
parser.add_argument("--save_last_dir", type = str, default = "./weights/slowfast_clip_21_dist_0_last.pt")
parser.add_argument("--save_result_dir", type = str, default = "./results/train_valid_loss_acc_slowfast_clip_21_dist_0.png")
parser.add_argument("--save_txt", type = str, default = "./results/test_slowfast_clip_21_dist_0.txt")
parser.add_argument("--save_conf", type = str, default = "./results/test_slowfast_clip_21_dist_0_confusion_matrix.png")
parser.add_argument("--use_focal_loss", type = bool, default = True)
parser.add_argument("--root_dir", type = str, default = "./dataset/dur21_dis0")

args = vars(parser.parse_args())

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:" + str(args["gpu_num"])
else:
    device = 'cpu'

# dataset composition
try:
    show_data_composition(args['root_dir'])
except:
    print("Directory is invalid")

if __name__ == "__main__":

    image_size = 224
    tau_fast = 1
    focal_gamma = 0.5
    alpha = args['alpha']
    mlp_hidden = args['hidden']
    batch_size = args['batch_size']
    lr = args['lr']
    seq_len = args['seq_len']
    num_epoch = args['num_epoch']
    verbose = args['verbose']
    gamma = args['gamma']
    save_best_dir = args['save_best_dir']
    save_last_dir = args['save_last_dir']
    save_conf = args["save_conf"]
    save_txt = args['save_txt']
    root_dir = args["root_dir"]
    image_size = args['image_size']

    train_data = CustomDataset(root_dir = root_dir, task = 'train', ts_data = None, augmentation = False, crop_size = image_size, seq_len = seq_len, mode = 'video')
    valid_data = CustomDataset(root_dir = root_dir, task = 'valid', ts_data = None, augmentation = False, crop_size = image_size, seq_len = seq_len, mode = 'video')
    test_data = CustomDataset(root_dir = root_dir, task = 'test', ts_data = None, augmentation = False, crop_size = image_size, seq_len = seq_len, mode = 'video')

    if args["use_sampler"]:
        train_sampler = ImbalancedDatasetSampler(train_data)
        valid_sampler = None
        test_sampler = None

    else:
        train_sampler = None
        valid_sampler = None
        test_sampler = None
    
    train_loader = DataLoader(train_data, batch_size = args['batch_size'], sampler=train_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    valid_loader = DataLoader(valid_data, batch_size = args['batch_size'], sampler=valid_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    test_loader = DataLoader(test_data, batch_size = args['batch_size'], sampler=test_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"])

    model = SlowFast(
        input_shape = (3, seq_len, image_size, image_size),
        block = Bottleneck3D,
        layers = [1,2,2,1],
        alpha = alpha,
        tau_fast = tau_fast,
        mlp_hidden = mlp_hidden,
        num_classes = 2,
        device = device
    )

    model.summary()

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=gamma)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma=gamma)

    if args['use_focal_loss']:
        train_data.get_num_per_cls()
        cls_num_list = train_data.get_cls_num_list()
        per_cls_weights = 1.0 / np.array(cls_num_list)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
        per_cls_weights = torch.FloatTensor(per_cls_weights)
        focal_gamma = 2.0
        loss_fn = FocalLoss(weight = per_cls_weights, gamma = focal_gamma)

    else: 
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "sum")

    train_loss,  train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        args['num_epoch'],
        args['verbose'],
        save_best_dir = save_best_dir,
        save_last_dir = save_last_dir,
        max_norm_grad = 1.0,
        criteria = "f1_score",
    )
    
    model.load_state_dict(torch.load(save_best_dir))

    # evaluation process
    test_loss, test_acc = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
        save_dir = args["save_test_result"]
    )