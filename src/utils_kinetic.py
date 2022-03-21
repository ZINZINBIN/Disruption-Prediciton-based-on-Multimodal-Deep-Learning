import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

video_height = 256
video_width = 256
video_channel = 3

# Training dataset for kinetics
kinetic_train_loader = torch.utils.data.DataLoader(
    datasets.Kinetics(
        root='.', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    ), batch_size=64, shuffle=True, num_workers=4
)

# Test dataset for mnist
kinetic_test_loader = torch.utils.data.DataLoader(
    datasets.Kinetics(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)