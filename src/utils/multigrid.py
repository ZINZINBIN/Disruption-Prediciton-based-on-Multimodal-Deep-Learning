import numpy as np
import logging
import torch
from torch.utils.data import sampler

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

class MultigridScheduler(object):
    def __init__(self):
        self.schedule = None

    def init_multigrid(self, cfg):
        cfg.MULTIGRID.DEFAULT_B = cfg.TRAIN.BATCH_SIZE
        cfg.MULTIGRID.DEFAULT_T = cfg.TRAIN.NUM_FRAMES
        cfg.MULTIGRID.DEFAULT_S = cfg.DATA.TRAIN_CROP_SIZE

        if cfg.MULTIGRID.LONG_CYCLE:
            self.schedule = self.get_long_cycle_schedule(cfg)
            cfg.SOLVER.STEPS = [0] + [s[-1] for s in self.schedule]
            cfg.SOLVER.STEPS[-1] = (
                cfg.SOLVER.STEPS[-2] + cfg.SOLVER.STEPS[-1]
            )

            # fine-tunning phase
            cfg.SOLVER.LRS = cfg.SOLVER.LRS[:-1] + [
                cfg.SOLVER.LRS[-2],
                cfg.SOLVER.LRS[-1]
            ]

            cfg.SOLVER.MAX_EPOCH = self.schedule[-1][-1]

        elif cfg.MULTIGRID.SHORT_CYCLE:
            cfg.SOLVER.STEPS = [
                int(s * cfg.MULTIGRID.EPOCH_FACTOR) for s in cfg.SOLVER.STEPS
            ]
            cfg.SOLVER.MAX_EPOCH = int(
                cfg.SOLVER.MAX_EPOCH * cfg.MULTIGRID.EPOCH_FACTOR
            )
        return cfg

    def get_long_cycle_schedule(self,cfg):
        steps = cfg.SOLVER.STEPS

        default_size = float(
            cfg.DATA.NUM_FRAMES * cfg.DATA.TRAIN_CROP_SIZE ** 2
        )
        default_iters = steps[-1]

        # Get shapes and average batch size for each long cycle shape.
        avg_bs = []
        all_shapes = []
        for t_factor, s_factor in cfg.MULTIGRID.LONG_CYCLE_FACTORS:
            base_t = int(round(cfg.DATA.NUM_FRAMES * t_factor))
            base_s = int(round(cfg.DATA.TRAIN_CROP_SIZE * s_factor))
            if cfg.MULTIGRID.SHORT_CYCLE:
                shapes = [
                    [
                        base_t,
                        cfg.MULTIGRID.DEFAULT_S
                        * cfg.MULTIGRID.SHORT_CYCLE_FACTORS[0],
                    ],
                    [
                        base_t,
                        cfg.MULTIGRID.DEFAULT_S
                        * cfg.MULTIGRID.SHORT_CYCLE_FACTORS[1],
                    ],
                    [base_t, base_s],
                ]
            else:
                shapes = [[base_t, base_s]]

            # (T, S) -> (B, T, S)
            shapes = [
                [int(round(default_size / (s[0] * s[1] * s[1]))), s[0], s[1]]
                for s in shapes
            ]
            avg_bs.append(np.mean([s[0] for s in shapes]))
            all_shapes.append(shapes)

        # Get schedule regardless of cfg.MULTIGRID.EPOCH_FACTOR.
        total_iters = 0
        schedule = []
        for step_index in range(len(steps) - 1):
            step_epochs = steps[step_index + 1] - steps[step_index]

            for long_cycle_index, shapes in enumerate(all_shapes):
                cur_epochs = (
                    step_epochs * avg_bs[long_cycle_index] / sum(avg_bs)
                )

                cur_iters = cur_epochs / avg_bs[long_cycle_index]
                total_iters += cur_iters
                schedule.append((step_index, shapes[-1], cur_epochs))

        iter_saving = default_iters / total_iters

        final_step_epochs = cfg.SOLVER.MAX_EPOCH - steps[-1]

        # We define the fine-tuning phase to have the same amount of iteration
        # saving as the rest of the training.
        ft_epochs = final_step_epochs / iter_saving * avg_bs[-1]

        schedule.append((step_index + 1, all_shapes[-1][2], ft_epochs))

        # Obtrain final schedule given desired cfg.MULTIGRID.EPOCH_FACTOR.
        x = (
            cfg.SOLVER.MAX_EPOCH
            * cfg.MULTIGRID.EPOCH_FACTOR
            / sum(s[-1] for s in schedule)
        )

        final_schedule = []
        total_epochs = 0
        for s in schedule:
            epochs = s[2] * x
            total_epochs += epochs
            final_schedule.append((s[0], s[1], int(round(total_epochs))))
        print_schedule(final_schedule)
        return final_schedule


    def update_long_cycle_schedule(self,cfg, cur_epoch):

        base_b, base_t, base_s = get_current_long_cycle_shape(
            self.schedule, cur_epoch
        )

        if base_s != cfg.DATA.TRAIN_CROP_SIZE or base_t != cfg.DATA.NUM_FRAMES:

            cfg.DATA.NUM_FRAMES = base_t
            cfg.DATA.TRAIN_CROP_SIZE = base_s
            cfg.TRAIN.BATCH_SIZE = base_b * cfg.MULTIGRID.DEFAULT_B

            bs_factor = (
                float(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
                / cfg.MULTIGRID.BN_BASE_SIZE
            )

            if bs_factor < 1:
                cfg.BN.NORM_TYPE = "sync_batchnorm"
                cfg.BN.NUM_SYNC_DEVICES = int(1.0 / bs_factor)
            elif bs_factor > 1:
                cfg.BN.NORM_TYPE = "sub_batchnorm"
                cfg.BN.NUM_SPLITS = int(bs_factor)
            else:
                cfg.BN.NORM_TYPE = "batchnorm"

            cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = cfg.DATA.SAMPLING_RATE * (
                cfg.MULTIGRID.DEFAULT_T // cfg.DATA.NUM_FRAMES
            )
            logger.info("Long cycle updates:")
            logger.info("\tBN.NORM_TYPE: {}".format(cfg.BN.NORM_TYPE))
            if cfg.BN.NORM_TYPE == "sync_batchnorm":
                logger.info(
                    "\tBN.NUM_SYNC_DEVICES: {}".format(cfg.BN.NUM_SYNC_DEVICES)
                )
            elif cfg.BN.NORM_TYPE == "sub_batchnorm":
                logger.info("\tBN.NUM_SPLITS: {}".format(cfg.BN.NUM_SPLITS))
            logger.info("\tTRAIN.BATCH_SIZE: {}".format(cfg.TRAIN.BATCH_SIZE))
            logger.info(
                "\tDATA.NUM_FRAMES x LONG_CYCLE_SAMPLING_RATE: {}x{}".format(
                    cfg.DATA.NUM_FRAMES, cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE
                )
            )
            logger.info(
                "\tDATA.TRAIN_CROP_SIZE: {}".format(cfg.DATA.TRAIN_CROP_SIZE)
            )
            return cfg, True
        else:
            return cfg, False


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

        if self.long_cycle_index in [0,1]:
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

def setup_data(
    dataset : Dataset,
    batch_size : int, 
    num_steps_per_update : int, 
    epochs : int, 
    iterations_per_epochs :int, 
    cur_iterations : int,
    crop_size : int,
    resize : int,
    num_frames : int,
    gamma_tau : float
    ):
    
    num_iterations = int(epochs * iterations_per_epochs)
    schedule = [int(i*num_iterations) for i in [0, 0.4, 0.65, 0.85, 1]]

    # transform function needed to change T,C,H,W -> T1,C1,H1,W1

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


def train_multigrid(
    train_dataset : Dataset,
    valid_dataset : Dataset,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn = None,
    init_lr : float = 0.001,
    warmup_steps : int = 8000,
    batch_size : int = BS * BS_UPSCALE,
    frames : int = 80,
    device : str = "cpu",
    num_epoch : int = 64,
    save_best_only : bool = False,
    save_best_dir : str = "./weights/best.pt"
    ):




    train_loss_list = []
    valid_loss_list = []
    
    train_acc_list = []
    valid_acc_list = []

    best_acc = 0
    best_epoch = 0
    best_loss = torch.inf

    for epoch in tqdm(range(num_epoch), desc = "training process"):

        train_loss, train_acc = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device 
        )

        valid_loss, valid_acc = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device 
        )

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