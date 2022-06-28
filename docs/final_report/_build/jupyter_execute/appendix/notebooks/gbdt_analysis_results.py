#!/usr/bin/env python
# coding: utf-8

# # Gradient Boosted Decision Tree (GBDT) models

# ## Key considerations
# 
# - **Data**: 
#     - Balanced dataset (B & C), containing data for muons(label = 0) and pions (label=1) with equal number of entries each.
#     - Unbalanced dataset (A)
# 
# - **Classifier type**: Binary 
# 
# 
# - **Features (3 Nos.)**:
# 
#     - `track_momentum`
#     
#     - `ring_radius`
#     
#     - `total_hits_filtered`
#     
# 
# - **Preprocessing**: 
# 
#     - Filtering noise in hits using delta = 0.3
#     - Preparing features and removing outlier or anomalous entries   
#     
# 
# - **Training**  
#     - Lightgbm, Xgboost, Adaboost for training with default values
#     - Final training and results on XGBoost on GPU
# 
# 
# - **Analysis and Results**    
#     - Analysis of model performance with respect to momentum bins
#     - Comparative ROC curves of XGBoost model on global and local momentum bins to assess potential bias    

# ## Data preparation
# 
# ### Train dataset (momentum balanced):
# - Balanced data set binned on 3 momentum bins:
#     - 15-25 GeV/C
#     - 25-35 GeV/C
#     - 35-45 GeV/C
#     
# ### Test datasets
# - balanced test set
# - unbalanced test set

# In[1]:


cd ../


# In[2]:


# Imports

# system libraries
import os
import sys
import glob

# modeling libraries
import pandas as pd
import numpy as np
import cudf
import cupy
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    plot_roc_curve,    
    auc, 
    accuracy_score,
    confusion_matrix, 
    classification_report,
    precision_recall_curve  
)
import optuna

# visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import h5py 
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# utils
from utils.helpers import *
from utils.gbt_dataset import *
from utils.plotting import *
logger = get_logger()
logger.setLevel(logging.CRITICAL)


# ### Checking GPU status

# In[3]:


get_ipython().system('nvidia-smi')


# In[4]:


# Global objects
random_state=123
gpu_id = get_config("gpu")[0]
cupy.cuda.Device(gpu_id).use()
print(f"gpu_id in use: {gpu_id}")


# In[5]:


get_ipython().run_cell_magic('time', '', '\n# Creating train dataset using gbt_df() util function\ndf_train = pd.DataFrame()\nfiles_ = get_config("dataset.train").items()\nfor dset_path_raw, dset_path_bal in files_:\n    df_ = gbt_df(dset_path_raw, dset_path_bal, gpu_id=gpu_id)\n    df_train = pd.concat([df_train, df_])\ndf_train\n')


# In[6]:


get_ipython().run_cell_magic('time', '', '\n# Creating test dataset - balanced using gbt_df() util function\ndf_test_bal = pd.DataFrame()\nfiles_ = get_config("dataset.test").items()\nfor dset_path_raw, dset_path_bal in files_:\n    df_ = gbt_df(dset_path_raw, dset_path_bal, gpu_id=gpu_id)\n    df_test_bal = pd.concat([df_test_bal, df_])\ndf_test_bal\n')


# In[7]:


# Test dataset - unbalanced
df_test_unbal = pd.DataFrame()
files_ = get_config("dataset.test").items()
for dset_path_raw, dset_path_bal in files_:
    df_ = gbt_df(dset_path_raw, dset_path_bal=None, gpu_id=gpu_id)
    df_test_unbal = pd.concat([df_test_unbal, df_])
df_test_unbal = df_test_unbal[df_test_unbal.label != 2] # Remove positrons
df_test_unbal


# In[8]:


# Summary of train test datasets
pd.DataFrame(
    {
        "df_train": df_train.label.value_counts(),
        "df_test_bal": df_test_bal.label.value_counts(),
        "df_test_unbal": df_test_unbal.label.value_counts()
    },    
)


# ## Training and evaluation on balanced data

# In[9]:


# X, y
X_train = df_train.loc[:, [
    'track_momentum',
    'ring_radius',
    'total_hits_filtered'
]]

y_train = df_train.loc[:,'label']

X_test_bal = df_test_bal.loc[:, [
    'track_momentum',
    'ring_radius',
    'total_hits_filtered'
]]
y_test_bal = df_test_bal.loc[:,'label']


# ### Default LightGBM classifier training and results

# In[10]:


get_ipython().run_cell_magic('time', '', 'lgbm_c = lgb.LGBMClassifier(random_state=random_state)\nlgbm_c.fit(X_train, y_train)\n')


# In[11]:


y_pred_lgb = lgbm_c.predict(X_test_bal)


# In[12]:


model_results(model=lgbm_c, X_test=X_test_bal, y_test=y_test_bal)


# ### XGBoost Classifier training and results

# In[13]:


get_ipython().run_cell_magic('time', '', 'xgb_cl = XGBClassifier(\n    random_state=random_state,\n    tree_method="gpu_hist",\n    gpu_id=gpu_id,\n    predictor="gpu_predictor",\n)\nxgb_cl.fit(X_train, y_train)\n')


# In[14]:


get_ipython().run_cell_magic('time', '', 'y_pred_xgb = xgb_cl.predict(X_test_bal)\n')


# In[15]:


model_results(xgb_cl, X_test_bal, y_test_bal)


# ### AdaBoost Classifier training and results

# In[16]:


get_ipython().run_cell_magic('time', '', 'ab_cl = AdaBoostClassifier(\n    random_state=random_state,\n    n_estimators=100\n)\nab_cl.fit(X_train, y_train)\n')


# In[17]:


y_pred_adb = ab_cl.predict(X_test_bal)


# In[18]:


model_results(ab_cl, X_test_bal, y_test_bal)


# In[19]:


# Comparing GBDT model performances on balanced test data across LGBM, XGBoost and AdaBoost
results_gbm = pd.DataFrame({
    "track_momentum": X_test_bal.track_momentum,
    "ring_radius": X_test_bal.ring_radius,
    "total_hits_filtered": X_test_bal.total_hits_filtered,
    "y_test": y_test_bal,
    "y_pred_lgb": y_pred_lgb,
    "y_pred_xgb": y_pred_xgb,
    "y_pred_adb": y_pred_adb,
    "compare_xgb_lgb": (y_pred_lgb == y_pred_xgb) 
    
})
results_gbm


# ## Analysis of different GBDT model predictions

# In[20]:


# Mismatch between lgbm and xgb predictions
print(f"Total mismatch between LBGM and XGB: {results_gbm[results_gbm.compare_xgb_lgb==False].shape[0]}")
results_gbm[results_gbm.compare_xgb_lgb==False]


# In[21]:


# Comparing xgboost predictions with actual labels
mis_preds = results_gbm.query('y_test != y_pred_xgb')
print(f"Total mis-predictions: {mis_preds.shape[0]}")
mis_preds


# ### Best performing model overall, and selecting one model going forward
# 
# Although, models performed similarly on the test data, XGBoost performance is better than the rest in terms of speed due to GPU accelaration support. Therefore, going forward, the GBDT model will be XGBClassifier with Sklearn API (native implementation is dmlc XGBoost)

# In[22]:


# XGBoost evaluation results on balanced test set for plotting function

y_probs_xgb = xgb_cl.predict_proba(X_test_bal)[:,1]
df_gbm = pd.DataFrame({
    "labels": y_test_bal,
    "predictions": y_pred_xgb,
    "probabilities": y_probs_xgb,
    "momentum": X_test_bal.track_momentum
    }
)
df_gbm


# ### Saving results df

# In[23]:


# Saving XGBoost results on balanced test set
path_preds_bal = '/fast_scratch_1/capstone_2022/models/xgboost/saved_models/xgb_balanced.csv'
df_gbm.to_csv(path_preds_bal)


# ### Plotting ROC curve

# In[24]:


# ROC curve at default (0.5) operating point
results = {
    "XGBoost_balanced_op_0.5": path_preds_bal
}
plot_roc_curves(
    results,
    title="ROC Curves: XGBoost models"
)


# ### Plotting Efficiency curves at different operating points

# In[25]:


file_path = '/fast_scratch_1/capstone_2022/models/xgboost/saved_models/xgb_balanced.csv'

operating_point = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
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
    
    title=f"XGBoost with Operating Point: {op}"
    
    df = pd.read_csv(file_path)
    df["predictions"] = np.where(df["probabilities"] > op, 1, 0)
    
    plot_cm(
        y_true=df["labels"],
        y_pred=df["predictions"],
        title=title
    )
    
    plot_efficiencies(
        path=file_path,
        title=title,
        cern_scale=True,
        pion_axlims=(0, 1),
        muon_axlims=(0, 45),
        pion_axticks=(0.05, 0.01),
        muon_axticks=(5, 1),
        op_point=op
)


# ### Choosing best operating point
# In order to balance the trade off between pion efficiencies (>0.75) and muon efficiencies(~ 0.01), 0.92 seems to be the best operating point for balanced test set.

# ### Efficiency plot of the best performing XGB model with operating point= 0.92

# In[26]:


file_path = '/fast_scratch_1/capstone_2022/models/xgboost/saved_models/xgb_balanced.csv'

operating_point = [0.92]

for op in operating_point:
    print("\033[1;33m" + "Operating point: " + str(op) + "\033[0m")
    print("\033[1;33m" + "-" * len("Operating point: " + str(op)) + "\033[0m")
    
    title=f"XGBoost with Operating Point: {op}"
    
    df = pd.read_csv(file_path)
    df["predictions"] = np.where(df["probabilities"] > op, 1, 0)
    
    plot = plot_efficiencies(
        path=file_path,
        title=title,
        cern_scale=True,
        pion_axlims=(0, 1),
        muon_axlims=(0, 45),
        pion_axticks=(0.05, 0.01),
        muon_axticks=(5, 1),
        save="docs/final_report/images/xgb_results_bal_0.92.svg",
        op_point=op
    )


# ## *Momentum Bias analysis*: Comparing models trained globally (15-45 GeV/c) and locally (15-25, 25-35 & 35-45 GeV/c)

# - Model: XGBClassifier
# - Train data: 0.75 split of balanced data (B and C)
# - Test data: 0.25 split of balanced data (B and C)

# In[27]:


get_ipython().run_cell_magic('time', '', '# Creating XGBoost model instance\nxgb_cl_ = XGBClassifier(\n    random_state=random_state,\n    tree_method="gpu_hist",\n    gpu_id=gpu_id,\n    predictor="gpu_predictor",\n)\n\n# Creating train test sets at 0.25 % split\nxgb_cl_15_25, X_train_1525, X_test_1525, y_train_1525, y_test_1525 = gbt_binwise(\n    model=xgb_cl_,\n    df_bal=df_train, \n    bin_low=15, \n    bin_high=25\n)\n')


# In[28]:


model_results(model = xgb_cl_15_25, X_test = X_test_1525, y_test = y_test_1525)


# In[29]:


get_ipython().run_cell_magic('time', '', '\n# Creating XGBoost model instance\nxgb_cl_ = XGBClassifier(\n    random_state=random_state,\n    tree_method="gpu_hist",\n    gpu_id=gpu_id,\n    predictor="gpu_predictor",\n)\n\nxgb_cl_25_35, X_train_2535, X_test_2535, y_train_2535, y_test_2535 = gbt_binwise(\n    model=xgb_cl_,\n    df_bal=df_train, \n    bin_low=25, \n    bin_high=35\n)\n')


# In[30]:


model_results(model = xgb_cl_25_35, X_test = X_test_2535, y_test = y_test_2535)


# In[31]:


get_ipython().run_cell_magic('time', '', '\n# Creating XGBoost model instance\nxgb_cl_ = XGBClassifier(\n    random_state=random_state,\n    tree_method="gpu_hist",\n    gpu_id=gpu_id,\n    predictor="gpu_predictor",\n)\n\nxgb_cl_35_45, X_train_3545, X_test_3545, y_train_3545, y_test_3545 = gbt_binwise(\n    model=xgb_cl_,\n    df_bal=df_train, \n    bin_low=35, \n    bin_high=45\n)\n')


# In[32]:


model_results(model = xgb_cl_35_45, X_test = X_test_3545, y_test = y_test_3545)


# ### ROC Curves

# In[33]:


# ROC Curves
fig = None
fig = plot_roc_curve(xgb_cl, X_test_bal, y_test_bal, name="XGBCl: 15-45")
fig = plot_roc_curve(xgb_cl_15_25, X_test_1525, y_test_1525, ax=fig.ax_, name="XGBCl: 15-25")
fig = plot_roc_curve(xgb_cl_25_35, X_test_2535, y_test_2535, ax = fig.ax_, name="XGBCl: 25-35")
fig = plot_roc_curve(xgb_cl_35_45, X_test_3545, y_test_3545, ax = fig.ax_, name="XGBCl: 35-45")
plt.title("ROC Curves for pions in different momentum bins")
plt.show()
plt.savefig("docs/final_report/images/xgb_results_pbins.svg")


# ## Evaluating results on 35-45 momentum bins with two different models: globally and locally trained xgboost models

# In[34]:


# ROC Curves
fig = None
# Globally trained model performance on 35-45 momentum bin
fig = plot_roc_curve(xgb_cl, X_test_3545, y_test_3545, name="XGBCl: 15-45 (global)")

# Locally trained model performance on 35-45 momentum bin
fig = plot_roc_curve(xgb_cl_35_45, X_test_3545, y_test_3545, ax = fig.ax_, name="XGBCl: 35-45 (local)")

plt.title(f"ROC Curves for pions in 35-45 momentum bins for sample size {X_test_3545.shape[0]}")
plt.show()
plt.savefig("docs/final_report/images/xgb_results_pbias_35_45.svg")


# ## Evaluating model on unbalanced test data

# In[35]:


# X, y
X_test_unbal = df_test_unbal.loc[:, [
    'track_momentum',
    'ring_radius',
    'total_hits_filtered'
]]
y_test_unbal = df_test_unbal.loc[:,'label']


# In[36]:


# Summary of test data
pd.DataFrame(
    {
        "df_train": df_train.label.value_counts(),
        "df_test_bal": df_test_bal.label.value_counts(),
        "df_test_unbal": df_test_unbal.label.value_counts()
    },    
)


# In[37]:


# preds
y_pred_xgb_unbal = xgb_cl.predict(X_test_unbal)


# In[38]:


# XGBoost evaluation results on balanced test set for plotting function
y_probs_xgb_unbal = xgb_cl.predict_proba(X_test_unbal)[:,1]
df_gbm_unbal = pd.DataFrame({
    "labels": y_test_unbal,
    "predictions": y_pred_xgb_unbal,
    "probabilities": y_probs_xgb_unbal,
    "momentum": X_test_unbal.track_momentum
    }
)
df_gbm_unbal


# ### Saving results df

# In[39]:


# Saving XGBoost results on balanced test set
path_preds_unbal = '/fast_scratch_1/capstone_2022/models/xgboost/saved_models/xgb_unbalanced.csv'
df_gbm_unbal.to_csv(path_preds_unbal)


# In[40]:


# ROC curve at default (0.5) operating point
results["XGBoost_unbalanced_op_0.5"] = path_preds_unbal
plot_roc_curves(
    results,
    title="ROC Curves: XGBoost models"
)


# ### Plotting Efficiency curves at different operating points

# In[41]:


file_path = '/fast_scratch_1/capstone_2022/models/xgboost/saved_models/xgb_unbalanced.csv'

operating_point = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
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
    
    title=f"XGBoost with Operating Point: {op}"
    
    df = pd.read_csv(file_path)
    df["predictions"] = np.where(df["probabilities"] > op, 1, 0)
    
    plot_cm(
        y_true=df["labels"],
        y_pred=df["predictions"],
        title=title
    )
    
    plot_efficiencies(
        path=file_path,
        title=title,
        cern_scale=True,
        pion_axlims=(0, 1),
        muon_axlims=(0, 45),
        pion_axticks=(0.05, 0.01),
        muon_axticks=(5, 1),
        op_point=op
)


# ## Observations on unbalanced test set performance

# As observed, XGBoost Classifier trained on balanced data and tested on unbalanced data is not performing well. Therefore, further analysis on XGBoost is not being carried out
