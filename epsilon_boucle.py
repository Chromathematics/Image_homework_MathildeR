# -*- coding: utf-8 -*-
"""


@author: rouxm
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import LambdaLR
from SqueezeNet import SqueezeNet
import pandas
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DEVICE = torch.device("cpu")

num_classes = 10  # 0-9

model = SqueezeNet(num_classes=num_classes).to(DEVICE)
#model.load_state_dict(torch.load("SqueezeNet_réduit_model_lr_e-3_epoch18_normalised0_1307_0_3081_batchsize_64.pth", map_location=DEVICE))
BATCH_SIZE = 16
EPSILON = 0 # Severness of the FGSM attack 
print("Setting Datasets") #OK
torch.manual_seed(42)  # for reproducibility

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


print("Downloading the MNIST database...")

train_ds = datasets.MNIST(root="./data", train = True,download=True,transform=transform['train']) #path ?
valid_ds = datasets.MNIST(root="./data",train= False, download =True,transform=transform['valid'])



train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
print("Data successfully downloaded !")
print("Classes:", train_ds.classes)

### FGSM attack
print(f"Disturbing image with an {EPSILON} severeness...")

def fgsm_attack(model, loss, images, labels, EPSILON) :
    EPS_NORM = EPSILON / 0.3081
    
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    images = images.clone().detach()
    images.requires_grad = True
    model.zero_grad()
    outputs = model(images)
    
    cost = loss(outputs, labels)
    cost.backward()
    
    #grad= images.grad.data
    attack_images = images + EPS_NORM*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1) #pas sûr
    if EPSILON == 0 : 
        return(images.detach())
    return attack_images.detach()

print(" Loading model...")
Path_to_pth = r"C://Users//rouxm//Desktop//2025-2026 - ARIA//Image//Homework//SqueezeNet_réduit_model_lr_e-3_epoch18_normalised0_1307_0_3081_batchsize_64.pth"
model.load_state_dict(torch.load(Path_to_pth, map_location=DEVICE))
loss = nn.CrossEntropyLoss()
model.eval()  # set to evaluation mode

dataframe =  pandas.read_excel(r"C://Users//rouxm//Desktop//2025-2026 - ARIA//Image//Homework//epsilon.xlsx", header=0)

for n in range(0,100): # to have integer

    EPSILON = 0.01*n
    print(f"{EPSILON} attack loading...")
    
    plt.clf()
    plt.figure(f"Confusion Matrix with $\epsilon$ = {EPSILON}",figsize=(400, 400))
    target=[]
    prediction=[]
    
    
    
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)
    CORRECT = 0
    TOTAL = 0
    for images, labels in valid_loader : 
        images = fgsm_attack(model, loss, images, labels, EPSILON).to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.no_grad():
            outputs = model(images)
            _,preds=torch.max(outputs,1)
            for label, pred in zip(labels, preds):
                total_per_class[label]+=1
                target.append(label)
                prediction.append(pred)
                TOTAL+=1
                if label == pred:
                    correct_per_class[label]+=1
                    CORRECT +=1
    acc_per_class = {}
    for i in range(num_classes):
        if total_per_class[i] > 0:
            acc_per_class[i] = 100.0 * correct_per_class[i].item() / total_per_class[i].item()
        else:
            acc_per_class[i] = None
    Liste = [EPSILON]
    cm= confusion_matrix(target,prediction)

    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(f"Normalisation_0_1307_0_3081//Confusion_matrix_eps_{EPSILON}_13_10_25.pdf")
    for cls, acc in acc_per_class.items():
        print(f"Class {cls} accuracy: {acc:.12f}% for $\epsilon$ = {EPSILON}")
        Liste.append(acc)
    Liste.append(100*CORRECT/TOTAL)
    dataframe.loc[len(dataframe)] = Liste
dataframe.to_excel("epsilon.xlsx", header=True,float_format="%.12f",index=False)
print(f"Correctness with $\epsilon$={EPSILON} saved in epsilon.xlsx")        


    

    
