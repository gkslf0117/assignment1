import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import random

# -------------------------------
# 1. 모델 정의 (CNN)
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = None  # adaptive
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# -------------------------------
# 2. 공격 함수들
# -------------------------------

# Targeted FGSM
def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    x_adv = x_adv - eps * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

# Untargeted FGSM
def fgsm_untargeted(model, x, label, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

# Targeted PGD
def pgd_targeted(model, x, target, k=40, eps=0.3, eps_step=0.01):
    k = int(k)
    x_adv = x.clone().detach()
    for _ in range(k):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv - eps_step * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv

# Untargeted PGD
def pgd_untargeted(model, x, label, k=40, eps=0.3, eps_step=0.01):
    k = int(k)
    x_adv = x.clone().detach()
    for _ in range(k):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv + eps_step * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv

# -------------------------------
# 3. 데이터셋
# -------------------------------
def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1, shuffle=False)
    return train_loader, test_loader

def get_cifar10_loaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform),
        batch_size=1, shuffle=False)
    return train_loader, test_loader


# -------------------------------
# 4. 학습 함수
# -------------------------------
def train(model, device, train_loader, epochs=15, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()




