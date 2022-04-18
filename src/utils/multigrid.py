import numpy as np
import os
import logging
import random
import torch
from tqdm import tqdm
from torch.utils.data import sampler
from src.dataloader import VideoDataset
from src.transforms.spatial_transforms import *
from src.transforms.temporal_transforms import *
from src.transforms.target_transforms import *
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader

# default value
BS = 8
BS_UPSCALE = 16 
INIT_LR = (1.6/1024)*(BS*BS_UPSCALE)
SCHEDULE_SCALE = 4
EPOCHS = (60000 * 1024 * 1.5)/220000 #(~420)

LONG_CYCLE = [8, 4, 2, 1]
LONG_CYCLE_LR_SCALE = [8, 0.5, 0.5, 0.5]
GPUS = 4
BASE_BS_PER_GPU = BS * BS_UPSCALE // GPUS 
CONST_BN_SIZE = 8

CROP_SIZE = {'S':112, 'M':224, 'XL':336}
RESIZE_SIZE = {'S':[128, 171], 'M':[256, 312], 'XL':[360, 420]}
GAMMA_TAU = {'S':6, 'M':5*2, 'XL':5}

logger = logging.getLogger(__name__)

def print_schedule(schedule):
    logger.info("Long cycle Index\tBase shape\tEpochs")
    for s in schedule:
        logger.info("{}\t{}\t{}".format(s[0], s[1], s[2]))

def get_current_long_cycle_shape(schedule, epoch):
    for s in schedule:
        if epoch < s[-1]:
            return s[1]
    return schedule[-1][1]

# step 1. 
# CycleBatchSampler
# iteration(epoch 포함)마다 index를 통해 (B, T, C, H, W) 의 형태를 결정
class RandomEpochSampler(sampler.RandomSampler):
    def __init__(self, data_source, replacement : bool = False, num_samples = None, epochs : int = 1):
        super(RandomEpochSampler, self).__init__(data_source, replacement, num_samples)
        self.epochs = epochs

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source) * self.epochs
        return self._num_samples * self.epochs
    
    def __len__(self):
        return self.num_samples
    
    def __iter__(self):
        n = len(self.data_source)

        while True:
            x =  torch.randperm(n).tolist()
            for v in x:
                yield v
        
class CycleBatchSampler(sampler.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, schedule, cur_iterations, long_cycle_bs_scale):
        super(CycleBatchSampler, self).__init__(sampler, batch_size, drop_last)
        self.schedule = schedule
        self.long_cycle_bs_scale = long_cycle_bs_scale

        self.iteration_counter = cur_iterations 
        self.short_iteration_counter = 0
        self.phase = 1
        self.phase_steps = ((self.schedule[self.phase] - self.schedule[self.phase - 1]) / len(self.long_cycle_bs_scale))
        self.long_cycle_index = 0
        self.iter_offset = 0

    def __iter__(self):
        batch_size = self.batch_size * self.long_cycle_bs_scale[self.long_cycle_index]
        self.short_iteration_counter = 0
        batch = []
        for _ in range(5):
            batch_size = self.adjust_long_cycle(batch_size)

        short_cycle_batch = self.adjust_short_cycle(batch_size)

        for idx in self.sampler:

            batch.append((idx, self.long_cycle_index))

            if len(batch) == short_cycle_batch:
                yield batch
            
                batch = []
                self.iteration_counter += 1
                self.short_iteration_counter += 1
                batch_size = self.adjust_long_cycle(batch_size)
                short_cycle_batch = self.adjust_short_cycle(batch_size)

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def adjust_long_cycle(self, batch_size):

        if self.iteration_counter > self.schedule[self.phase]:
            self.iter_offset = self.schedule[self.phase]
            self.phase += 1
            self.phase_steps = ((self.schedule[self.phase] - self.schedule[self.phase - 1]) / len(self.long_cycle_bs_scale))
            self.long_cycle_index = 0

            if self.phase == len(self.schedule) - 1:
                self.long_cycle_index = -1
            
            batch_size = (self.batch_size * self.long_cycle_bs_scale[self.long_cycle_index])
        
        elif self.iteration_counter >= self.phase_steps + self.iter_offset:
            self.iter_offset += self.phase_steps
            self.long_cycle_index += 1

            if self.phase == len(self.schedule) - 1:
                self.long_cycle_index = -1
            
            self.long_cycle_index = min(self.long_cycle_index, len(self.long_cycle_bs_scale) - 1)
            batch_size = (self.batch_size * self.long_cycle_bs_scale[self.long_cycle_index])

        return batch_size

    def adjust_short_cycle(self, batch_size):

        if self.long_cycle_index in [0, 1]:
            if self.short_iteration_counter % 2 == 0:
                short_cycle_batch = batch_size * 2
            if self.short_iteration_counter % 2 == 1:
                short_cycle_batch = batch_size
        else:
            if self.short_iteration_counter % 3 == 0:
                short_cycle_batch = batch_size * 4
            if self.short_iteration_counter % 3 == 1:
                short_cycle_batch = batch_size * 2
            if self.short_iteration_counter % 3 == 2:
                short_cycle_batch = batch_size

        return short_cycle_batch

# step 2.
# Dataset
# iteration index에 따라 각기 다른 (B,T,C,H,W)를 Crop하는 역할
class MultigridDataset(VideoDataset):
    def __init__(
        self, dataset = "fast_model_dataset", split = "test", 
        clip_len = 16, preprocess = False, augmentation : bool = True, 
        multigrid : bool = True, temporal_transform = None, spatial_transform = None, target_transform = None,
        sample_duration : int = 16, crop_size : int = 224):
        super(MultigridDataset, self).__init__(dataset, split, clip_len, preprocess, augmentation)
        self.multigrid = multigrid
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.target_transform = target_transform
        self.crop_size = crop_size
        self.sample_duration = sample_duration
        
        self.long_cycles = [
            (self.sample_duration//4, int(np.floor(self.crop_size / np.sqrt(2)))),
            (self.sample_duration//2, int(np.floor(self.crop_size / np.sqrt(2)))),
            (self.sample_duration//2, self.crop_size),
            (self.sample_duration, self.crop_size)
        ]
        
    def __getitem__(self, index):
        
        iteration = index[0]
        index, long_cycle_state = index[1]
        
        sample_duration, crop_size = self.long_cycles[long_cycle_state]
        stats = (sample_duration, crop_size //2, int(np.floor(crop_size / np.sqrt(2))), crop_size)
        
        if long_cycle_state in [0,1]:
            short_cycle_state = iteration % 2
            
            if short_cycle_state == 0:
                crop_size = int(np.floor(crop_size / np.sqrt(2)))
            
        else:
            short_cycle_state = iteration % 3
            if short_cycle_state == 0:
                crop_size = crop_size // 2
            elif short_cycle_state ==1:
                crop_size = int(np.floor(crop_size / np.sqrt(2)))
                
        buffer = self.load_frames(self.fnames[index])

        if buffer.shape[0] < self.clip_len :
            buffer = self.load_frames(self.fnames[index-(self.clip_len - buffer.shape[0])])

        if buffer.shape[0] < self.clip_len:
            buffer = self.refill_temporal_slide(buffer)

        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == "train" and self.augmentation:
            buffer = self.brightness(buffer, val = 30, p = 0.25)
            buffer = self.contrast(buffer, 1, 1.5, p = 0.25)
            buffer = self.blur(buffer, p = 0.25, kernel_size = 5)
            buffer = self.randomflip(buffer, p = 0.25)
            buffer = self.vertical_shift(buffer, ratio = 0.2, p = 0.25)
            buffer = self.horizontal_shift(buffer, ratio = 0.2, p = 0.25)

        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        
        frame_indices = [idx for idx in range(len(buffer))]
        
        if self.temporal_transform is not None:
            t_stride = random.randint(1, max(1, self.sample_duration // sample_duration))
            frame_indices = self.temporal_transform(frame_indices, t_stride, sample_duration)

        buffer = buffer[frame_indices]
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters(crop_size)
            buffer = self.spatial_transform(buffer)
        
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        
        return torch.from_numpy(buffer), torch.from_numpy(labels), long_cycle_state, stats
    
def get_dataset(
    dataset = "fast_model_dataset", split = "test", 
    clip_len = 16, preprocess = False, augmentation : bool = True, 
    multigrid : bool = True, temporal_transform = None, spatial_transform = None, target_transform = None,
    sample_duration : int = 16, crop_size : int = 224):
    
    dataset = MultigridDataset(dataset, split, clip_len, preprocess, augmentation, multigrid, temporal_transform, spatial_transform, target_transform, sample_duration, crop_size)
    return dataset

# step 3.
# train code
# sampler에 index를 순서대로 호출하여 그에 맞는 입력값을 model에 집어넣어 학습 진행
from typing import List

def setup_data(
    dataset : Dataset,
    batch_size : int, 
    num_steps_per_update : int, 
    epochs : int, 
    iterations_per_epochs :int, 
    cur_iterations : int,
    crop_size : int,
    resize : List[int],
    num_frames : int,
    gamma_tau : float
    ):
    
    num_iterations = int(epochs * iterations_per_epochs)
    schedule = [int(i*num_iterations) for i in [0, 0.4, 0.65, 0.85, 1]]

    # transform function needed to change T,C,H,W -> T1,C1,H1,W1
    
    train_transforms = {
        "spatial" : Compose([
            MultiScaleRandomCropMultigrid([crop_size / i for i in resize], crop_size)
        ]),
        "temporal": TemporalRandomCrop(num_frames, gamma_tau),
        "target":ClassLabel()
    }
    
    drop_last = False
    shuffle = True
    
    if shuffle:
        sampler = RandomEpochSampler(dataset, epochs = epochs)
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = CycleBatchSampler(sampler, batch_size, drop_last, schedule=schedule, cur_iterations=cur_iterations, long_cycle_bs_scale=LONG_CYCLE)
    dataloader = DataLoader(dataset, num_workers = 4, batch_sampler=batch_sampler, pin_memory=True)

    schedule[-2] = (schedule[-2] + schedule[-1]) // 2

    return dataloader, dataset, schedule[1:]

def train_multigrid(
    train_dataset : Dataset,
    valid_dataset : Dataset,
    model : torch.nn.Module,
    init_lr : float = 0.001,
    warmup_steps : int = 8000,
    batch_size : int = BS * BS_UPSCALE,
    frames : int = 80,
    crop_size : int = 128,
    resize : List[int] = [180, 224],
    gamma_tau : float = 6,
    verbose : int = 4,
    device : str = "cpu",
    num_epoch : int = 64,
    save_best_only : bool = False,
    save_best_dir : str = "./weights/best.pt"
    ):
    
    st_steps = 204000
    load_steps = 204000
    steps = 204000
    num_steps_per_update = 1
    iterations_per_epochs = train_dataset.__len__() // batch_size
    val_iterations_per_epochs = valid_dataset.__len__() // batch_size
    cur_iterations = steps * num_steps_per_update
    max_steps = iterations_per_epochs * num_epoch
    num_frames = 0
    
    train_dataloader, train_dataset, lr_schedule = setup_data(train_dataset, batch_size, num_steps_per_update, num_epoch, iterations_per_epochs, cur_iterations, crop_size, resize, num_frames, gamma_tau)
    
    lr_schedule = [i // num_steps_per_update for i in lr_schedule]
    

    RESTART = False
    
    if steps > 0:
        load_ckpt = torch.load(save_best_dir)
        cur_long_ind = load_ckpt['long_ind']
        bn_splits = model.update_bn_splits_long_cycle(LONG_CYCLE[cur_long_ind])
        model.load_state_dict(load_ckpt['model_state_dict'])
        last_long = cur_long_ind
        RESTART = True
        
    
    lr = init_lr
    print("Initial learning rate : {}".format(lr))
    
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=5e-5)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule)
    
    if steps > 0:
        optimizer.load_state_dict(load_ckpt['optimizer_state_dict'])
        lr_sched.load_state_dict(load_ckpt['scheduler_state_dict'])    
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    
    train_loss_list = []
    valid_loss_list = []
    
    train_acc_list = []
    valid_acc_list = []

    best_acc = 0
    best_epoch = 0
    best_loss = torch.inf

    for epoch in tqdm(range(num_epoch), desc = "training process"):
        # train process
        train_loss = 0
        train_acc = 0
        
        model.train()
        for batch_idx, (data, target, long_ind, stats) in enumerate(train_dataloader):
            optimizer.zero_grad()
            long_ind = long_ind[0].item()
            
            if long_ind != last_long:
                bn_splits = model.update_bn_splits_long_cycle(LONG_CYCLE[long_ind])
                lr_scale_fact = LONG_CYCLE[long_ind] if (last_long == -2 or long_ind == -1) else LONG_CYCLE_LR_SCALE[long_ind]
                last_long = long_ind
                
                for g in optimizer.param_groups:
                    g['lr'] *= lr_scale_fact
                    lr = g['lr']
                
            elif RESTART:
                RESTART = False
                
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            train_acc += pred.eq(target.view_as(pred)).sum().item() / data.size(0) 

        
            lr_warmup(lr, steps - st_steps, warmup_steps, optimizer)
            steps += 1
            lr_sched.step()
            
        train_loss /= (batch_idx + 1)
        train_acc /= (batch_idx + 1)
        
        # valid process
        valid_loss = 0
        valid_acc = 0
        
        model.eval()
        for batch_idx, (data, target, long_ind, stats) in enumerate(valid_dataloader):
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


def lr_warmup(init_lr : float, cur_steps : int, warmup_steps : int, opt = None):

    start_after = 1
    if cur_steps < warmup_steps and cur_steps > start_after:
        lr_scale = min(1., float(cur_steps +1) / warmup_steps)

        for pg in opt.param_groups:
            pg['lr'] = lr_scale * init_lr

def print_stats(long_ind, batch_size, stats, gamma_tau, bn_splits, lr):
    bs = batch_size * LONG_CYCLE[long_ind]

    if long_ind in [0,1]:
        bs = [bs * j for j in [2,1]]
        print(' ***** LR {} Frames {}/{} BS ({},{}) W/H ({},{}) BN_splits {} long_ind {} *****'.format(lr, stats[0][0], gamma_tau, bs[0], bs[1], stats[2][0], stats[3][0], bn_splits, long_ind))
    else:
        bs = [bs*j for j in [4,2,1]]
        print(' ***** LR {} Frames {}/{} BS ({},{},{}) W/H ({},{},{}) BN_splits {} long_ind {} *****'.format(lr, stats[0][0], gamma_tau, bs[0], bs[1], bs[2], stats[1][0], stats[2][0], stats[3][0], bn_splits, long_ind))
