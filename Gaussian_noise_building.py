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
import numpy as np

class AddGaussianNoise(object):
    def __init__(self,mean=0.1,std=0.1):
        self.std = std
        self.mean = mean
    def __call__(self,tensor):
        return(tensor + torch.randn(tensor.size())*self.std + self.mean)
    def __repr__(self):
        return(self.__class__.__name__+ '(mean={0}, std={1})'.format(self.mean, self.std))
DEVICE = torch.device("cpu")

num_classes = 10  # 0-9
model = SqueezeNet(num_classes=num_classes).to(DEVICE)
print(" Loading model...")
#Path_to_pth = r"C://Users//rouxm//Desktop//2025-2026 - ARIA//Image//Homework//SqueezeNet_réduit_model_lr_e-3_epoch20.pth"
Path_to_pth = r"C:\Users\rouxm\Desktop\2025-2026 - ARIA\Image\Homework\SqueezeNet_advtrained_eps0.070_advfrac0.5_epochs18.pth"
Path_to_pth=r"C:\Users\rouxm\Desktop\2025-2026 - ARIA\Image\Homework\SqueezeNet_réduit_model_lr_e-3_epoch18_normalised0_1307_0_3081_batchsize_64.pth"
model.load_state_dict(torch.load(Path_to_pth, map_location=DEVICE))
BATCH_SIZE = 16
print("Setting Datasets") #OK
torch.manual_seed(42)  # for reproducibility


m=0
std=0
dataframe=pandas.DataFrame({'m':[],'std' :[],'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[],'total':[],})
for i in range(50):
    m=0.01*i
    for j in range(20):
        std=0.01*j
        print(f"exploring mean {m} and std {std}")
        
        Liste=[m,std]
        # Datasets
        transform = {
            'train': transforms.Compose([ #il faut verouiller son ordi sinon je fais un rm -rf / :) :) 
                transforms.ToTensor(),
                AddGaussianNoise(m,std),
                transforms.Normalize([0.1307],[0.3081])
                ]),
            'valid': transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(m,std),
                transforms.Normalize([0.1307],[0.3081])
                ])
            }       

        print("Downloading the MNIST database with gaussian noise...")

        train_ds = datasets.MNIST(root="./data", train = True,download=True,transform=transform['train']) #path ?
        valid_ds = datasets.MNIST(root="./data",train= False,download =True,transform=transform['valid'])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
        print("Data successfully downloaded !")
        print("Classes:", train_ds.classes)
        loss = nn.CrossEntropyLoss()
        model.eval()  # set to evaluation mode
        correct_per_class = torch.zeros(num_classes)
        total_per_class = torch.zeros(num_classes)
        CORRECT = 0
        TOTAL = 0
        for images,label in valid_loader :
            images= images.to(DEVICE)
            label=label.to(DEVICE)
            with torch.no_grad():
                outputs=model(images)
                _,preds=torch.max(outputs,1)
                for label, pred in zip(label,preds):
                    total_per_class[label]+=1
                    TOTAL+=1
                    if label == pred:
                        correct_per_class[label]+=1
                        CORRECT+=1
        acc_per_class = {}
        for i in range(num_classes):
            if total_per_class[i] > 0:
                acc_per_class[i] = 100.0 * correct_per_class[i].item() / total_per_class[i].item()
            else:
                acc_per_class[i] = None
        for cls, acc in acc_per_class.items():
            print(f"Class {cls} accuracy: {acc:.12f}%")
            Liste.append(acc)
        Liste.append(100*CORRECT/TOTAL)
        dataframe.loc[len(dataframe)] = Liste
        print(f"Accuracy with gaussian noise m={m}, std= {std} = {100*CORRECT/TOTAL:.2f}%")
dataframe.to_excel("gaussian_noise_adversarial_trained.xlsx",header=True,index=False,float_format="%.12f")
print('Done')






















