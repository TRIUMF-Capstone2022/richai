# 3. Data Science Methods

## 3.1: Baseline model: XGBoost

### 3.1.1: Why we used this model

XGBoost {cite}`Chen_2016` was chosen as the baseline machine learning model due to it's scalability and ease of use with tabular data.  In addition, XGBoost offers support for GPU acceleration, which is ideal when working with a large dataset.  The benefits of using XGBoost as our baseline model are further discussed in {ref}`App. C.1 <appendix:ml_gbm:intro>`. 

### 3.1.2: Features

The fitted ring radius, particle momentum, and total number of photon light hits (filtered for hits with a time close to the time the particle entered the detector), were used as features to train the XGBoost models.  Note that the point cloud conversion was not performed for the XGBoost baseline model.

### 3.1.3: Model benefits and shortcomings

In terms of benefits, XGBoost is easy, simple and intuitive to use.  Also, XGBoost is very efficient in terms of training time due to GPU acceleration.

In terms of shortcomings, the final XGBoost model does not capture the actual coordinate position of the photon light hits, as this feature is a variable in length (each event has a different number of light hits), and XGBoost is not designed to support variable length features. Further, the final XGBoost model did not perform well for pion or muon efficiency relative to CERN's prior algorithm.

### 3.1.4: Results

```{figure} ../images/xgb_results_bal_0.92.svg
---
name: xgb_results
---

XGBoost results
```

As observed above in {numref}`xgb_results`, the pion efficiency drops sharply as particle momentum increases beyond 35 GeV/C. In addition, the muon efficiency is poor across most momentum bins.

```{figure} ../images/xgb_results_pbins.svg
---
name: ROC_xgboost
---

ROC curves of XGBoost models on different momentum bins
```

A global XGBoost model was trained over the full 15-45 GeV/c particle momentum range and it was observed that the XGBoost model performed poorly in higher momentum bins as shown below in {numref}`ROC_xgboost` above.

### 3.1.4: Justification for moving onto deep learning

The main factor that drove the decision to explore deep learning models was the poor performance of XGBoost on particles with high momentum values.  Further, XGBoost could not handle variable length photon hit data, whereas deep learning models can.  Ultimately, to improve the results from the baseline performance, a more complex model was needed.

## 3.2: Deep learning

### 3.2.1: Selecting models

In selecting the deep learning models to apply, the focus was on finding models that were specifically designed to work with point cloud data. Two models were identified with architectures that appeared to fit the problem scope well.  The first model that was identified is called [PointNet](https://arxiv.org/abs/1612.00593) {cite}`qi2017pointnet`, and was developed by researchers at Stanford University.  The second model that was identified is called [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829) {cite}`wang2019dynamic`, and was developed by researchers at Massachusetts Institute of Technology, UC Berkeley, and Imperial College London.

### 3.2.2: Features 

Two separate feature combinations were tested when tuning both PointNet and the Dynamic Graph CNN model.  The first feature combination included the point cloud coordinates, momentum, and radius data generated from CERN's prior algorithm as input.  The second feature combination was a simpler model that did not take the radius as input, but still received the point cloud information and momentum data. Comparing the results from both these models will reveal if the radius is needed as a feature, or if the point cloud and momentum data is enough for significant performance.

{ref}`App. D.2 <appendix:deeplearning:pointnet:hyp>` and {ref}`App. E.2 <appendix:deeplearning:dgcnn:hyp>` detail the hyperparameters tuned for PointNet and Dynamic Graph CNN respectively. 

### 3.2.3: PointNet

#### Model architecture 

The overall architecture for the model used in this analysis is detailed in {ref}`App. D.1 <appendix:deeplearning:pointnet:arch>`.

#### Model benefits and shortcomings

In terms of benefits, PointNet was able to achieve a strong pion efficiency across all momentum bins, while maintaining similar muon efficiency to prior NA62 performance.  Further,  PointNet uses a symmetric function (max pooling) to make it robust to any change in the order of the coordinates that make up the input point cloud data.  Also, PointNet uses a Spatial Transformer Network {cite}`jaderberg2015spatial` to make it robust to any spatial variability within the input point cloud data

In terms of shortcomings, PointNet required the longest training time of all the models we tested (~24 hours on three GPUs), and, by it's design, PointNet does not capture local information from within the coordinate point cloud input data.

#### Best performing model

```{figure} ../images/pointnet_roc.png
:name: pointnet_roc

PointNet ROC Curves
```

As can be seen in the receiver operating characteristic curves ("ROC") in {numref}`pointnet_roc`, the best PointNet model used all of the available features (hits coordinate point cloud, momentum, and radius).  Further, in order to filter photon hits based on the time delta of photon light hit time less the time the particle entered the detector, a value of 0.50ns worked best.  The learning rate that was used was a linear annealing scheduler, which started at an initial learning rate of 0.0003 and increased to 0.003 in the first half of training, before decreasing back to the original learning rate in the second half of training to help convergence.  The total number of epochs of training was 16.

Finally, an operating point of 0.93 was selected on the ROC curve, as this allowed the PointNet model to achieve a strong pion efficiency while also maintaining good muon efficiency relative to the prior NA62 results.

### 3.2.4: Dynamic Graph CNN

#### Model architecture 

The overall architecture for the model used in this analysis is  detailed in {ref}`App. E.1 <appendix:deeplearning:dgcnn:arch>`. 

#### Model benefits and shortcomings

In terms of benefits, Dynamic Graph CNN is designed to capture local information within the point cloud data {cite}`wang2019dynamic`, which is something that PointNet cannot do.  Further, Dynamic Graph CNN total training time was almost two times faster than PointNet, as the number of model parameters was less.

In terms of shortcomings, our best Dynamic Graph CNN can maintain a similar pion efficiency to PointNet, however, it struggles with muon efficiency.  Dynamic Graph CNN does not make use of a spatial transformer network, and therefore is not resistant to rotations of the input point cloud data.

#### Best performing model

```{figure} ../images/dgcnn_roc.png
:name: dgcnn_roc

Dynamic Graph CNN ROC Curves
```

As can be seen in the receiver operating characteristic curves ("ROC") in {numref}`dgcnn_roc`, the best Dynamic Graph CNN model also used all of the features.  The time delta that worked best for filtering the photon light hits was 0.30ns.  The learning rate used was the same as PointNet.  The total number of epochs trained for was 8.

An operating point of 0.96 on the ROC curve was selected, as this allowed the model to achieve a strong pion efficiency.  An operating point could not be found that achieved both a good pion efficiency and a good muon efficiency.

## 3.3: Overall model results

### 3.3.1: Selecting the overall best model

```{figure} ../images/all_models_roc.png
:name: all_models_roc

ROC Curves: All models
```

As can be seen in {numref}`all_models_roc`, the overall best model was PointNet, following by Dynamic Graph CNN.  Both deep learning models were able to surpass our baseline XGBoost model.

### 3.3.2: Deep learning model efficiencies

```{figure} ../images/dgcnn_efficiency.png
:name: dgcnn_efficiency

Dynamic Graph CNN Model Efficiencies
```

```{figure} ../images/pointnet_efficiency.png
:name: pointnet_efficiency

PointNet Graph CNN Model Efficiencies
```

In analyzing the final model performance in terms of pion and muon efficiency in {numref}`dgcnn_efficiency` and {numref}`pointnet_efficiency`, it can be seen that both the PointNet and Dynamic Graph CNN models were able to surpass the prior NA62 pion efficiency performance across all momentum bins.  Unfortunately, the Dynamic Graph CNN model was unable to achieve a similar muon efficiency to NA62 in any momentum bin.  However, the PointNet model was able to surpass NA62 muon efficiency in momentum bins greater than 34 GeV/c.
