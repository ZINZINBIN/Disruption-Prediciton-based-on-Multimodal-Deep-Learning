from src.models.model import SlowFastDisruptionClassifier
from src.models.resnet import Bottleneck3D
from src.utils.multigrid import train_multigrid
import torch

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

    alpha = 2
    hidden = 128
    p = 0.5

    model = SlowFastDisruptionClassifier(
        input_shape = (3,42,112,112),
        block = Bottleneck3D,
        layers = [1,2,2,1], 
        alpha = alpha,
        p = p,
        mlp_hidden = hidden,
        num_classes  = 2,
        base_bn_splits=8
    )

    train_multigrid(
        model, 
        "dur0.2_dis21", 
        init_lr = 0.001, warmup_steps=8000, num_epoch=64, 
        save_best_only=True, save_best_dir = "./weights/multigrid_best.pt",
        device = device
        )