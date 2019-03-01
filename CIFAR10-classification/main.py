import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import AllCNNModel, BasicBlock, ResidualBlock 
from trainer import train_loop, test
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_type", required=True)
parser.add_argument("--n_conv", required=True, type=int, help="Number of additional conv used")
parser.add_argument("--n_epochs", required=True, type=int, help="Number of epochs used to run the experiments")

options = parser.parse_args()

assert(options.experiment_type in ["basic", "resnet"]), "basic or resnet only!!"

if options.experiment_type == "basic":
    blocktype = BasicBlock
else:
    blocktype = ResidualBlock

exp_name = f"{options.experiment_type}_{options.n_conv}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create directories to store model and results
os.makedirs(f'./results/{exp_name}', exist_ok=True)
os.makedirs(f'./models/{exp_name}', exist_ok=True)


# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

n_total = len(trainset)
n_valid = 10000
n_train = n_total - n_valid

train_data, valid_data = torch.utils.data.random_split(trainset, (n_train, n_valid))

trainloader = torch.utils.data.DataLoader(
    train_data, batch_size=64, shuffle=True, num_workers=2
)

validloader = torch.utils.data.DataLoader(
    valid_data, batch_size=64, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2
)


def test_forward(net):
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


n_epoch = options.n_epochs

net = AllCNNModel(blocktype, 5, options.n_conv)
test_forward(net)
model = net.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=0.001)

train_loop(model, optimizer, criterion, trainloader, validloader, device, n_epoch, exp_name)

model.load_state_dict(torch.load(f'./models/{exp_name}/best.net'))

test(model, testloader, device, exp_name)

