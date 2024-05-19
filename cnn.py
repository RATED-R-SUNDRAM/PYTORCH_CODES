#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torchvision
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

