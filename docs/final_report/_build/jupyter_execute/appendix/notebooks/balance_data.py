#!/usr/bin/env python
# coding: utf-8

# # Creating balanced data

# - **Purpose**: The purpose of this notebook is to demonstrate the balanced data generation process.
# - **Author**: Shiv Jena
# - **module used**: `dataset.balance_data.py`

# In[1]:


cd ../


# In[2]:


# importing module
from dataset.balance_data import *
from utils.helpers import get_config


# ## Changes in `configs/config.yaml` to create new save paths
# 
# The original source path is the key in the `train` unit, and the save path is the value. In order to securely create balanced dataset without overwriting the original data, it is suggested that the save path be changed as shown below.
# >```
#     dataset:
#           delta: 0.5 #pointnet: 0.5, dgcnn: 0.3
#     train:
#           /data/bvelghe/capstone2022/B/2018B.Sample.EOSlist.CTRL.p.v2.0.4_f.v2.0.4_patched.h5:
#               /fast_scratch_1/capstone_2022/combined_datasets/B.2018B_copy.h5
#           /data/bvelghe/capstone2022/C/2018E.EOSlist.CTRL_patched.h5:
#               /fast_scratch_1/capstone_2022/datasetC_combined_copy.h5
# >```

# In[3]:


get_ipython().run_cell_magic('time', '', '# Creating and saving balanced data using balance_data()\nbalance_data(dset_path = get_config("dataset.train").items())\n')


# In[4]:


# defining original and sample file paths for matching
path_original_b = '/fast_scratch_1/capstone_2022/combined_datasets/B.2018B.h5'
path_sample_b = '/fast_scratch_1/capstone_2022/combined_datasets/B.2018B_copy.h5'
path_original_c = '/fast_scratch_1/capstone_2022/datasetC_combined.h5'
path_sample_c = '/fast_scratch_1/capstone_2022/datasetC_combined.h5'


# ## Testing for Original and Sample B

# In[5]:


# Comparing original and sample files of B dataset
df_b_org = pd.read_hdf(path_original_b)
df_b_sample = pd.read_hdf(path_sample_b)


# In[6]:


df_b_org


# In[7]:


df_b_sample


# In[8]:


# Comparing sample and original for label==1 (pions)
df_b_sample_pions = df_b_sample.query('label==1')
df_b_org_pions = df_b_org.query('label==1')
df_b_sample_pions.equals(df_b_org_pions)


# **Conclusion**: This shows two datasets are equal for pion entries. Muon entries could also be equal, if generated with random seed.

# ## Testing for Original and Sample C

# In[9]:


# Comparing original and sample files of B dataset
df_c_org = pd.read_hdf(path_original_c)
df_c_sample = pd.read_hdf(path_sample_c)


# In[10]:


df_c_org


# In[11]:


df_c_sample


# In[12]:


# Comparing both dataframes
df_c_sample.equals(df_c_org)


# **Conclusion**: This shows two datasets are identical.
