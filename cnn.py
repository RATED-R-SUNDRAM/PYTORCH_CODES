#%%
from tkinter.ttk import _Padding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow_code
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torchvision
import torch.nn.functional as F
#%%
import torch
print(torch.cuda.is_available())
# %%
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_data = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
# %%
print(type(train_set))
print(type(train_data))
print(dir(train_set))
print(dir(test_data))

# %%
# BASIC DATA VISUALIZATION
images,labels = next(iter(train_data))
print(type(labels),type(images))
print(labels.shape,images.shape)
print(images[0].shape)
plt.imshow(images[0].squeeze())
print(labels[0])
# %%

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*25, 128)
        self.fc2 = nn.Linear(128, 10)
        self.conv3 = nn.Linear(10,1,)

    def forward(self, x):
        x = F.relu(self.conv1(x))# (28-3)+1 (26,26)
        x = self.maxpool(x) # (26//2) (13,13)
        x = F.relu(self.conv2(x)) # (11,11)
        x = self.maxpool(x)#(5,5)
        x = x.reshape(x.shape[0], -1) # (16*5*5)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

model = CNN()
optimizer= torch.optim.SGD(model.parameters(),lr=0.001)
loss= nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# %%


#%%
epoch=0 
flag=True
while(flag):
    for i,(images,labels) in enumerate(train_data):
        output = model(images)
        loss_value = loss(output,labels)
        print("loss_value =",loss_value.item())
        if(loss_value.item()<0.4):
            flag=False
            break
        optimizer.zero_grad()
        loss_value.backward()

        optimizer.step()
        if(i%100==0):
            print(f"Epoch {epoch} Batch {i} Loss {loss_value.item()}")
        
    lr_scheduler.step()
    epoch+=1

# %%

tot=0
cnt=0
with torch.no_grad():
    for i,(images,labels) in enumerate(test_data):
        output = model(images)
        _,predicted = torch.max(output,1)
        tot+=64
        #print(_,predicted)
        cnt+= (predicted==labels).sum().item()
print(cnt/tot)

        
# %%

print(output.shape)
print(_.shape)
print(predicted.shape)
print(output)
print(_)
print(predicted)
# %%
torch.save(model.state_dict(), "cnn_model.pth")
# %%
