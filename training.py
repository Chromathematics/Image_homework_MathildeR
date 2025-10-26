# -*- coding: utf-8 -*-
"""


@author: rouxm
"""

from SqueezeNet import SqueezeNet   

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import os

# Parameters
print("Charging Parameters")

DATA_DIR = r"C://Users//rouxm//Desktop//2025-2026 - ARIA//Image//SqueezeNet"
BATCH_SIZE = 16
EPOCHS = 18
LR = 1e-3
LR_START=1e-3
LR_END=1e-4

DEVICE = torch.device("cpu")

print("Setting Datasets") #OK

# Datasets
transform = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307],[0.3081])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307],[0.3081])
    ])
}


train_ds = datasets.MNIST(root="./data", train = False,download=True,transform=transform['train']) #path ?
print("train_ds created") #Tourne, mais après si il le considère vide ...
valid_ds = datasets.MNIST(root="./data",train= False, download =True,transform=transform['valid'])



train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
print("Train_loader Successfully created") #NON ! --> trainLoader marche pas
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_ds.classes)

# Model
model = SqueezeNet(num_classes=len(train_ds.classes)).to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

lr_lambda = lambda epoch: (LR_END/LR_START) + (1 - epoch/(EPOCHS-1)) * (1 - LR_END/LR_START)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS} ",f"lr={optimizer.param_groups[0]['lr']:.6f}")
    # ---- Train ----
    model.train()
    running_loss, correct, total = 0, 0, 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f"Train loss: {running_loss/total:.4f}, acc: {correct/total:.4f}")

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Valid acc: {correct/total:.4f}")
    

torch.save(model.state_dict(), "SqueezeNet_réduit_model_lr_e-3_epoch18_normalised0_1307_0_3081_batchsize_64.pth")
print("Model saved!")
