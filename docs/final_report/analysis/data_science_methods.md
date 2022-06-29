# 3. Data Science Methods

## 3.1: Baseline model: Gradient boosted trees

### 3.1.1: Why we used this model

XGBoost {cite}`Chen_2016` was chosen as our baseline machine learning model due to it's scalability and ease of use with tabular data.  In addition, XGBoost offers support for GPU acceleration, which was ideal in our scenario of working with a large dataset.  In general, the benefits of using XGBoost as our baseline model are discussed in {ref}`App. C.1 <appendix:ml_gbm:intro>`. 

### 3.1.2: Features

We used the following features in training our XGBoost models:
- `Ring radius`: Radius of the the circle fitted by TRIUMF's prior algorithm to the photon light hits of the particle decay.
- `Momentum`: Particle momentum information.
- `Total hits`: An engineered feature on total number of photon hits per decay event.  The "hits" were filtered to keep only those close in time, where "closeness" was determined based on a delta between the photon hit time and the CHOD detector time.

### 3.1.3: Model benefits and shortcomings

In terms of benefits, XGBoost is easy, simple and intuitive to use.  XGBoost is efficient in terms of low training time via GPU acceleration.

In terms of shortcomings, our XGBoost model does not capture the coordinate position data of the photon hits, as this is a variable length feature and XGBoost cannot support this without significant feature engineering.  Further, XGBoost did not perform well for pion or muon efficiency.

### 3.1.4: Results

```{figure} ../images/xgb_results_bal_0.92.svg
---
name: xgb_results
---

XGBoost results
```

As observed above in {numref}`xgb_results`, pion efficiency drops sharply with increase in momentum beyond 35 GeV/C. In addition, muon efficiency is also poor.

Further, different XGBoost models were trained and tested on different momentum bins. A Global XGBoost model trained over the 15-45 GeV/c momentum bin, as well as local xgboost models were trained and evaluated. It was observed that the models performed poorly in higher momentum bins as shown below in {numref}`ROC_xgboost`.

```{figure} ../images/xgb_results_pbins.svg
---
name: ROC_xgboost
---

ROC curves of XGBoost models on different momentum bins
```

### 3.1.4: Justification for moving onto deep learning

The main factor that drove our decision to apply deep learning models was the poor performance of XGBoost on particles with high momentum.  Further, XGBoost could not handle variable length photon hit data, whereas deep learning models can.  Ultimately, to improve our results from the baseline, we needed a more complex model that could extract features directly and more precisely from the photon hit position data, in addition to also identifying non-linear relationships between these features with particle momentum and ring radius.  Deep learning models were a good fit for this, and hence our decision to pursue these more complex models.

## 3.2: Deep learning

### 3.2.1: Selecting models

In selecting our deep learning models, we searched for models that were specifically designed to work well with point cloud coordinate data.  Based on this research, we identified two models with architectures that appeared to fit our problem scope well.  The first model that we identified was called [PointNet](https://arxiv.org/abs/1612.00593) {cite}`qi2017pointnet`, which was developed by researchers at Stanford University.  The second model that we identified was called [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829) {cite}`wang2019dynamic`, which was developed by researchers at Massachusetts Institute of Technology, UC Berkeley, and Imperial College London.  We also note that traditional convolutional neural networks would not work well for our data due to its sparsity.

### 3.2.2: Features 

Two separate feature combinations were tested when tuning both PointNet and the Dynamic Graph CNN model.  The first feature combination included the photon hit coordinates point cloud, momentum, and radius data generated from the MLE algorithm as input.  The second feature combination was a simpler model that did not take the radius as input, but still received the hits point cloud and momentum data. Comparing the results from both these models will reveal how effectively the deep learning architecture extracts the radius information from the point clouds alone, and if the abstraction of this data (ring radius value) is required at all.

{ref}`App. D.2 <appendix:deeplearning:pointnet:hyp>` and {ref}`App. E.2 <appendix:deeplearning:dgcnn:hyp>` detail the hyperparameters tuned for PointNet and Dynamic Graph CNN respectively. 

### 3.2.3: PointNet

#### Model architecture 

The overall architecture for the model used in this analysis, including the modifiactions made to the base implementation, is  detailed in {ref}`App. D.1 <appendix:deeplearning:pointnet:arch>`.

#### Model benefits and shortcomings

In terms of benefits, PointNet was able to achieve a strong pion efficiency across all momentum bins, while maintaining similar muon efficiency to prior NA62 performance.  Further,  PointNet uses a symmetric function (max pooling) to make it robust to any change in the order of the coordinates that make up the input point cloud data.  Also, PointNet uses a Spatial Transformer Network {cite}`jaderberg2015spatial` to make it robust to any spatial variability within the input point cloud data

In terms of shortcomings, PointNet required the longest training time of all the models we tested (~24 hours on three GPUs), and, by it's design, PointNet does not capture local information from within the coordinate point cloud input data.

#### Best performing model

```{figure} ../images/pointnet_roc.png
:name: pointnet_roc

PointNet ROC Curves
```
As can be seen in the receiver operating characteristic curves ("ROC") in {numref}`pointnet_roc`, our best PointNet model used all of the available features (hits coordinate point cloud, momentum, and radius).  Further, in order to filter photon hits based on a time delta of hit time - CHOD time, we found that 0.20ns worked best for our model.  The learning rate that we used was a linear annealing scheduler, which starts with an initial learning rate of 0.0003 and increases to 0.003 in the first half of training, before decreasing back to the original learning rate in the second half of training.  The total number of epochs trained for were 16.

Finally, we selected an operating point of 0.93 on the ROC curve, as this allowed us to achieve a strong pion efficiency while maintaining good muon efficiency relative to the NA62 results.

### 3.2.4: Dynamic Graph CNN

#### Model architecture 

The overall architecture for the model used in this analysis, including the modifiactions made to the base implementation, is  detailed in {ref}`App. E.1 <appendix:deeplearning:dgcnn:arch>`. 

#### Model benefits and shortcomings

In terms of benefits, Dynamic Graph CNN is designed to capture local information within the point cloud data {cite}`wang2019dynamic`, which is something that our other model (PointNet) cannot do.  Further, Dynamic Graph CNN total training time was almost two times faster than PointNet, as the number of model parameters was less.

In terms of shortcomings, our best Dynamic Graph CNN can maintain a similar pion efficiency to PointNet, however, it struggles with muon efficiency.  We also note that Dynamic Graph CNN does not make use of a spatial transformer network, and therefore is not resistant to rotations of the input point cloud data.

#### Best performing model

```{figure} ../images/dgcnn_roc.png
:name: dgcnn_roc

Dynamic Graph CNN ROC Curves
```

As can be seen in the receiver operating characteristic curves ("ROC") in {numref}`dgcnn_roc`, our best Dynamic Graph CNN model also used all of the features.  The time delta that worked best for filtering our hits was 0.30ns.  The learing rate used was the same as PointNet.  The total number of epochs trained for was 8.

We selected an operating point of 0.96 on the ROC curve, as this allowed us to achieve a strong pion efficiency.  We were unable to select an operating point that achieved both a good pion efficiency and a good muon efficiency.

## 3.3: Overall model results

### 3.3.1: Selecting the overall best model

```{figure} ../images/all_models_roc.png
:name: all_models_roc

ROC Curves: All models
```

In selecting our overall best model, we used the ROC curve.  As can be seen in the {numref}`all_models_roc`, our overall best model was PointNet, following by Dynamic Graph CNN.  Both deep learning models were able to surpass our baseline XGBoost model.

### 3.3.2: Model efficiencies

```{figure} ../images/dgcnn_efficiency.png
:name: dgcnn_efficiency

Dynamic Graph CNN Model Efficiencies
```

```{figure} ../images/pointnet_efficiency.png
:name: pointnet_efficiency

PointNet Graph CNN Model Efficiencies
```

In analyzing our final model performance in terms of pion and muon efficiency in {numref}`dgcnn_efficiency` and {numref}`pointnet_efficiency`, we note that both the PointNet and Dynamic Graph CNN models were able to surpass the prior NA62 pion efficiency performance across all momentum bins.  Unfortunately, the Dynamic Graph CNN model was unable to achieve a similar muon efficiency to NA62 in any momentum bin.  Finally, the PointNet model was able to surpass NA62 muon efficiency in momentum bins > 34 GeV/c.
