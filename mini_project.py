#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import torch
import torch.nn as nn

#%% 
from PIL import Image
img= np.array(Image.open('image.jpg'))