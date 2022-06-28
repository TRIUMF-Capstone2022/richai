#!/usr/bin/env python
# coding: utf-8

# # Dynamic graph CNN operating point

# - **Purpose**: The purpose of this notebook is to select an operating point for the Dynamic Graph CNN model.
# - **Author**: Nico Van den Hooff

# In[1]:


cd ../


# In[2]:


import pandas as pd
import numpy as np
from utils.plotting import plot_efficiencies, plot_cm
from utils.helpers import get_config


# # Results for operating points

# In[3]:


path = "/fast_scratch_1/capstone_2022/models/dgcnn/dgcnn_predictions.csv"

operating_point = [
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
    
    title=f"DGCNN with Operating Point: {op}"
    
    df = pd.read_csv(path)
    df["predictions"] = np.where(df["probabilities"] > op, 1, 0)
    
    plot_cm(
        y_true=df["labels"],
        y_pred=df["predictions"],
        title=title
    )
    
    plot_efficiencies(
        path=path,
        title=title,
        cern_scale=True,
        pion_axlims=(0, 1),
        muon_axlims=(0, 45),
        pion_axticks=(0.05, 0.01),
        muon_axticks=(5, 1),
        op_point=op
)

