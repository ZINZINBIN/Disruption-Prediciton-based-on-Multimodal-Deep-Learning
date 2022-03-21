import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Training dataset for mnist
mnist_train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='.', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    ), batch_size=64, shuffle=True, num_workers=4
)

# Test dataset for mnist
mnist_test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)

video_height = 256
video_width = 256
video_channel = 3

# Training dataset for kinetics
kinetic_train_loader = torch.utils.data.DataLoader(
    datasets.Kinetics(
        root='.', split = 'train', download=True,
        frames_per_clip = 6,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    ), batch_size=64, shuffle=True, num_workers=4
)

# Test dataset for mnist
kinetic_test_loader = torch.utils.data.DataLoader(
    datasets.Kinetics(
        root='.', split = 'test', download=True, 
        frames_per_clip = 6,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)


# Training dataset for UCF101
ucf_train_loader = torch.utils.data.DataLoader(
    datasets.UCF101(
        root='.', split = 'train', download=True,
        frames_per_clip = 6,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    ), batch_size=64, shuffle=True, num_workers=4
)

# Test dataset for UCF101
ucf_test_loader = torch.utils.data.DataLoader(
    datasets.UCF101(
        root='.', split = 'test', download=True, 
        frames_per_clip = 6,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)




def convert_image_np(inp:torch.Tensor):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn(train_loader : torch.utils.data.DataLoader, test_loader : torch.utils.data.DataLoader, model : torch.nn.Module, device : str = 'cpu'):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

        plt.savefig(f, "./results/mnist_model_results.png")