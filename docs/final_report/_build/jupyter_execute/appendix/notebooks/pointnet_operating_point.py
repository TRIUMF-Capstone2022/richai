#!/usr/bin/env python
# coding: utf-8

# # PointNet Operating point 

# - **Purpose**: The purpose of this notebook is to select an operating point for the PointNet model.
# - **Model Used**: `/fast_scratch_1/capstone_2022/models/pointnet/saved_models/pointnet_momentum_radius_delta_0.5_16e_0706.pt` 
# - **Features**: Event POS, momentum and radius
# - **Epochs**: 16
# - **Time delta**: 0.5ns 

# In[1]:


# model path
path = "/fast_scratch_1/capstone_2022/models/pointnet/saved_models/pointnet_momentum_radius_delta_0.5_16e_0706_unbalanced_p_15_45.csv"


# In[2]:


cd ../


# # Imports

# In[3]:


import pandas as pd
import numpy as np
from utils.plotting import plot_efficiencies, plot_cm
from utils.helpers import get_config


# # Results for operating points

# In[4]:


operating_point = [
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
]

for op in operating_point:
    print("\033[1;33m" + "Operating point: " + str(op) + "\033[0m")
    print("\033[1;33m" + "-" * len("Operating point: " + str(op)) + "\033[0m")

    title = f"PointNet with Operating Point: {op}"

    df = pd.read_csv(path)
    df["predictions"] = np.where(df["probabilities"] > op, 1, 0)

    plot_cm(y_true=df["labels"], y_pred=df["predictions"], title=title)

    plot_efficiencies(
        path=path,
        title=title,
        cern_scale=True,
        pion_axlims=(0, 1),
        muon_axlims=(0, 45),
        pion_axticks=(0.05, 0.01),
        muon_axticks=(5, 1),
        op_point=op,
    )


# In[ ]:




