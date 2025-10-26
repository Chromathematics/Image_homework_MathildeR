# -*- coding: utf-8 -*-
"""


@author: rouxm
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from SqueezeNet import SqueezeNet 

class FGSM_Attack(object):
    def __init__(self, eps=0.1, mean=0.1307, std=0.3081, device='cpu'):

        self.eps = eps
        self.mean = mean
        self.std = std
        self.device = device

        # normalized min/max for clamping after perturbation:
        # if original pixel in [0,1], normalized = (x-mean)/std
        self.min_norm = (0.0 - self.mean) / self.std
        self.max_norm = (1.0 - self.mean) / self.std

    def __call__(self, model, X, y, loss_fn):

        # ensure we do not modify original tensor in-place
        X_adv = X.detach().clone().to(self.device).requires_grad_(True)
        model.eval()   # eval mode for stable gradients (no dropout)
        outputs = model(X_adv)
        loss = loss_fn(outputs, y.to(self.device))
        model.zero_grad()
        loss.backward()

        # sign of gradient
        grad_sign = X_adv.grad.data.sign()
        # apply perturbation in normalized space
        X_adv = X_adv + self.eps * grad_sign

        # clamp to normalized valid range (so denormalized values are in [0,1])
        X_adv = torch.clamp(X_adv, self.min_norm, self.max_norm).detach()

        model.train()
        return X_adv

    def __repr__(self):
        return f"FGSM_Attack(eps={self.eps}, mean={self.mean}, std={self.std}, device={self.device})"

# -------------------------
# Parameters & dataset
# -------------------------
DATA_DIR = r"C://Users//rouxm//Desktop//2025-2026 - ARIA//Image//SqueezeNet"
BATCH_SIZE = 16
EPOCHS = 18
LR = 1e-3
LR_START = 1e-3
LR_END = 1e-4

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

transform = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])
}

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform['train'])
valid_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform['valid'])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Train / Valid sizes:", len(train_ds), len(valid_ds))
print("Classes:", train_ds.classes)

model = SqueezeNet(num_classes=len(train_ds.classes)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_lambda = lambda epoch: (LR_END / LR_START) + (1 - epoch/(EPOCHS-1)) * (1 - LR_END / LR_START)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


eps = 0.07              
adv_frac = 0.6         
use_adversarial = True  
attacker = FGSM_Attack(eps=eps, mean=mean, std=std, device=DEVICE)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS} lr={optimizer.param_groups[0]['lr']:.6f}")
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

       
        if use_adversarial and adv_frac > 0:
            batch_size = X.size(0)
            
            mask = torch.rand(batch_size, device=DEVICE) < adv_frac
            if mask.any():

                X_adv_full = attacker(model, X, y, criterion)  
                X = X.clone()
                X[mask] = X_adv_full[mask]

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")

    # validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for Xv, yv in valid_loader:
            Xv, yv = Xv.to(DEVICE), yv.to(DEVICE)
            outputs = model(Xv)
            _, preds = torch.max(outputs, 1)
            correct += (preds == yv).sum().item()
            total += yv.size(0)
    val_acc = correct / total
    print(f"Valid acc: {val_acc:.4f}")

# Save the model
save_name = "SqueezeNet_advtrained_eps{:.3f}_advfrac{}_epochs{}.pth".format(eps, adv_frac, EPOCHS)
torch.save(model.state_dict(), save_name)
print("Model saved to:", save_name)
