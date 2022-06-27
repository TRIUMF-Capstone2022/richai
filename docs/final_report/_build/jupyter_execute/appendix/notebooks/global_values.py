#!/usr/bin/env python
# coding: utf-8

# # Global values for debiasing and standardization
# **Purpose**: The purpose of this notebook is to calculate:
# - Global mean $(x, y)$ ring centre locations to use in debiasing the data
# - Global mean and standard deviation values for momentum to be used in standardizing this feature before it is fed into the neural networks
# - Global mean and standard deviation values for ring radius to be used in standardizing this feature before it is fed into the neural networks
# 
# **Author**: Nico Van den Hooff

# In[1]:


import pandas as pd
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


# # Unfiltered data

# In[2]:


df = pd.read_hdf("/fast_scratch_1/capstone_2022/datasetC_combined.h5")


# In[3]:


df.describe().round(3).T


# # Filtered data

# In[4]:


filtered_df = df.query("ring_radius < 500 and ring_radius > 0")
filtered_df = filtered_df.query("ring_centre_pos_x < 2500 and ring_centre_pos_x > -2500")
filtered_df = filtered_df.query("ring_centre_pos_y < 2500 and ring_centre_pos_y > -2500")


# In[5]:


filtered_df.describe().round(3).T


# In[6]:


print("Before removing outliers:")
print("Ring centre mean x:", df["ring_centre_pos_x"].mean())
print("Ring centre mean y:", df["ring_centre_pos_y"].mean())
print("Momentum mean:", df["track_momentum"].mean())
print("Momentum std:", df["track_momentum"].std())
print("Ring radii mean:", df["ring_radius"].mean())
print("Ring radii std:", df["ring_radius"].std())


# In[7]:


print("After removing outliers:")
print("Ring centre mean x:", filtered_df["ring_centre_pos_x"].mean())
print("Ring centre mean y:", filtered_df["ring_centre_pos_y"].mean())
print("Momentum mean:", filtered_df["track_momentum"].mean())
print("Momentum std:", filtered_df["track_momentum"].std())
print("Ring radii mean:", filtered_df["ring_radius"].mean())
print("Ring radii std:", filtered_df["ring_radius"].std())

