#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow_code
import torch
import cv2
import torch.nn as nn

#%% 
from PIL import Image
img_array = cv2.imread("./9.jpg", cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(img_array, (28, 28))
plt.imshow(resized_image)
plt.show()
# %%
print(resized_image.shape)

# %%

model.load_state_dict(torch.load("cnn_model.pth"))
x= torch.tensor(resized_image, dtype=torch.float32).reshape(1,1,28,28)
# %%
print(model(x))
_,predictions = torch.max(model(x),1)
print(predictions)
# %%

# %%

# %%
