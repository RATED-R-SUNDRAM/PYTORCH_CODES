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
df['target'] = df['target'].replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
x=torch.tensor(df.iloc[:,0:4].values,dtype=torch.float32)
y=torch.tensor(np.eye(3)[list(df['target'])],dtype=torch.float32)
print(y)
# %%

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.fc1=nn.Linear(4,3)
        self.relu=nn.Sigmoid()
        self.fc2=nn.Linear(3,3)
        self.relu2 = nn.Sigmoid()
        self.output=nn.Softmax()
    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.output(x)
        return x

loss =nn.CrossEntropyLoss()
mod = model()
optim = torch.optim.Adam(mod.parameters(),lr=0.05) # type: ignore

epoch = 0

while(epoch <10000):
    y_pred =mod(x)
   # print(y_pred)
    l=loss(y_pred,y)
    optim.zero_grad() 
    l.backward()
    optim.step() 
    
    if(epoch%1000==0):
        print(y[0],y_pred[0])
        print("loss :",l.item())
        print()
    epoch+=1
#%% accuracy check 

y_pred = torch.argmax(mod(x),dim=1)
y_= torch.argmax(y,dim=1)
print(y_pred,y_)
cnt=0
for i in range(y_.shape[0]):
    if y_[i]==y_pred[i]:
        cnt+=1 
print(cnt/y_.shape[0])


        

# %%
