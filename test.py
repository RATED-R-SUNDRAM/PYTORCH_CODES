#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import torch
import torch.nn as nn

#%% 
df =pd.read_csv(r"D:\ONE_DRIVE\Desktop\iris_dataset.csv")
df.head()
df['target'] = df['target'].replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
x= torch.tensor(df.drop(['target'],axis=1).values,dtype=torch.float32)
#y= torch.tensor(df['target'].values,dtype=torch.int)
#print(x.shape,y.shape)
y_new= np.eye(3)[list(df['target'])]
y_new= torch.tensor(y_new,dtype=torch.float32)
print(y_new)

#y= y.unsqueeze(1)
#%%
class Linear(nn.Module):
    def __init__(self,in_features,out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features,out_features)
    def forward(self,x):
        return torch.sigmoid(self.linear(x))

model = Linear(4,3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    y_pred = model(x)
    loss = criterion(y_pred, y_new)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        #print(y_pred,y_new)
        print(f'Epoch {epoch+1}, Loss = {loss.item():.4f}')

# %%
[w,b]=model.parameters()
print(w[0],b)
# %%
