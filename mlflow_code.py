#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow_code
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 

#%%
import mlflow
#mlflow.create_experiment("test_experiment")
mlflow.set_experiment("test_experiment")
mlflow.end_run()
mlflow.start_run(run_name="run_final2")
mlflow.log_param("param1", "value1")


#%% load data and convert to tensor 
mlflow.sklearn.autolog()
df= pd.read_csv(r"D:\ONE_DRIVE\Desktop\PYTORCH\iris.csv")
df['species']=df['species'].astype('category')
df['species_cat']=df['species'].cat.codes
#print(df.head)
print(df.columns)
df=df.drop(['species'],axis=1)
print(df.columns)
x= np.array(df.iloc[:,:-1])
y= np.array(df.iloc[:,-1])
X_train, X_test, y_train, y_test = train_test_split(x,y , 
                                   random_state=104,  
                                   test_size=0.25,  
                                   shuffle=True) 
xt= torch.from_numpy(x).float()
yt=  torch.from_numpy(y).long()
print(x,y)
#%% 
clf = LogisticRegression(random_state=0).fit(X_train, y_train) 
print(clf.score(X_test, y_test))
#print(clf.([[32.0,29.1,23,9]], [[1]]))
# %%
