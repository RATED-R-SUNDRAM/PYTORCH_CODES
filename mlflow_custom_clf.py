#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split 
#%% 
#mlflow.create_experiment("custom_clf_pytorch")
mlflow.set_experiment("custom_clf_pytorch")
mlflow.end_run()
mlflow.start_run(run_name="custom_clf_pytorch3")
#%%
mlflow.autolog()
df= pd.read_csv(r"D:\ONE_DRIVE\Desktop\PYTORCH\iris.csv")
df['species']=df['species'].astype('category')
df['species_cat']=df['species'].cat.codes
#print(df.head)
print(df.columns)
df=df.drop(['species'],axis=1)
print(df.columns)
x= np.array(df.iloc[:,:-1])
y= np.array(df.iloc[:,-1])
y=np.eye(3)[y]
X_train, X_test, y_train, y_test = train_test_split(x,y , 
                                   random_state=104,  
                                   test_size=0.25,  
                                   shuffle=True) 
xtr= torch.tensor(X_train,dtype=torch.float)
ytr=  torch.tensor(y_train,dtype=torch.float)
xte= torch.tensor(X_test,dtype=torch.float)
yte=  torch.tensor(y_test,dtype=torch.float)
print(x,y)
#%%
print(xtr.shape,ytr.shape,xte.shape,yte.shape)
#%%
print(ytr)
#%%

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 6)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = Model()
criterion = nn.BCELoss()
optim=torch.optim.RMSprop(model.parameters(),lr=0.05)
#%%

class custom(mlflow.pyfunc.PythonModel):
    def __init__(self,model,loss,optim):
        super(custom,self).__init__()
        self.model=model
        self.loss=loss
        self.optim=optim
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def fit(self,X_train,y_train):
        epoch = 0
        while(epoch<1000):
            y_pred = self.model(X_train)
            loss = self.loss(y_pred,y_train)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            epoch+=1
            if(epoch%100==0):
                print("Epoch:",epoch,"Loss:",loss.item()) 
    def predict(self,X):
        return self.model(X)  # Adjusted to accept only X
    def configure_optimizers(self):
        return self.optim
    def get_model(self):
        return self.model
    
clf = custom(model,criterion,optim)
clf.fit(xtr,ytr)

# %%
mlflow.pyfunc.log_model(artifact_path="model",python_model=clf)
# %%

clf2 = mlflow.pyfunc.load_model("runs:/a8fc04526db8406f8778e06487a749b1/model")
#%%
clf2 = clf2.unwrap_python_model()
# %%
clf2.predict(xte)
# %%
print(type(clf2))
original_model = custom(clf2.params)
# %%
