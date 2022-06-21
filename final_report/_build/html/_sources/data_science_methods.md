<!-- #region -->
# Data Science Methods

## Baseline model: Gradient boosted trees
### Why GBT are a good baseline model?
Gradient Boosted Decision Trees (GDBT) use ensemble of decision trees sequentially minimising a loss function and hence, are popular due their efficiency, accuracy and ability to avoid overfitting. Besides, the libraries associated offer flexibility in terms of parameters such as decision tree algorithm, loss function, regularization, GPU related parameters etc. which make them a popular first choice as baseline models. 

There are several algorithm based GBDTs available on open source platforms such as Lightgbm, Catboost, Xgboost, Adaboost etc. They are mostly available in form of individual libraries with native implementations and with sklearn in some cases.

### Why we used this model
As the number of features were low in our case, the decision tree algorithms were not expected to vary considerably. Therefore, Xgboost (XGBClassifier) with sklearn API was chosen as our baseline GBDT for benchmarking purposes primarily due to its support in form of [parameters](https://xgboost.readthedocs.io/en/stable/gpu/index.html) enabling GPU accelaration for faster training.


### Features that were used for GBT
The following features were used for Xgboost:
- ring_radius: radius of the the circle fitted by the MLE algorithm (provided by TRIUMF)
- track_momentum: momentum data as provided along with data (provided by TRIUMF)
- total_hits_filtered: an engineered feature on total number of hits per event/entry, filtered with a fixed value of delta (chod_time - hit_time)

### Pros/Cons
Pros: 
- simple and intuitive, as the features used are physics informed properties of the particles (classes)
- efficient in terms of low training time
Cons:
- does not capture the position data of the hits as it directly captures the MLE radius
- takes a fixed value of delta (chod_time - hit_time) without any scope to learn how to eliminate noise in the data
- low pion efficiency after limiting the muon efficiency

### Results
![](images/xgb_results.png)

As observed above, the pion efficiency drops sharply with increase in momentum beyond 35 GeV/C. Besides, muon efficiency is poor at the chosen operating point.
Further, different xgboost models were trained and tested on different momentum bins. A Global xgboost model trained over 15-45 GeV/c momentum bin as well as local xgboost models were trained and evaluated. It was observed that the models performed poorly in higher momentum bins as shown below:
![](images/ROC_xgboost.png)

The following ROC curves plot establishes that the models were actually leveraging discriminating power of input features and not biased by distributional issues in data.
![](images/ROC_xgboost_3545.png)

### Justification for moving onto deep learning
 - Xgboost GBDT model did not use position data of the hits. Instead, it used the already engineered feature - ring_radius from the analytical MLE method leaving no scope for improving the results. 
- Therefore, to improve results, more accurate models were required which could extract features directly and more precisely from the hits position data.

Thus, deep learning models were which leverage feature extraction from position data in form of point clouds comprised the further steps in modeling approach beyond baseline GBDT model. 


## Deep learning

### Selecting models

In selecting our deep learning models, we searched for models that were specifically designed to work well with point cloud coordinate data.  Based on this research, we identified two models with architectures that appeared to fit our problem scope well.  The first model that we identified was called [PointNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) {cite}`qi2017pointnet`, which was developed by researchers at Stanford University.  The second model that we identified was called Dynamic Graph CNN {cite}`wang2019dynamic`, which was developed by researchers at Massachusetts Institute of Technology, UC Berkely, and Imperial College London.  We also note that traditional convolutional neural networks would not work well for our data due to its sparsity.

### Tuning models

We tried out two separate feature combinations when tuning both PointNet and the Dynamic Graph CNN models.  The first feature combination was a model that took the hits point cloud, momentum, and radius data as input to the neural networks.  The second feature combination was a simpler model that did not take the radius as input, but still received the hits point cloud and momentum data.

In terms of hyperparameters, the common hyperparameters that were tuned were the:

- Time delta between the hit time and the CHOD time, for which values of 0.20ns to 0.50ns were used
- Learning rate, for which a constant learning rate and a learning rate scheduler were used
- Number of epochs, for which a maximum of 24 epochs were used

The Dynamic Graph CNN had an additional hyperparameter, which was the value $k$ for the K-nearest neighbours graph that is dynamically generated by the models architecture.  We tried values between 8 and 20 nearest neighbors.

### PointNet

#### Model benefits

- PointNet achieves a strong pion efficiency accross all momentum bins, while maintaining similar muon efficiency to prior NA62 performance
- PointNet uses a symmetric function (max pooling) to make it robust to any change in the order of the coordinates that make up the input point cloud data
- PointNet uses a Spatial Transformer Network {cite}`jaderberg2015spatial` to make it robust to any spatial variability within the input point cloud data

#### Model shortcomings

- PointNet requires the longest training time of all the models we tested (~24 hours on three GPUs)
- By it's design, PointNet is not able to capture local information within the point cloud

#### Best performing model

![pointnet_roc](images/pointnet_roc.png)

As can be seen in the receiver operating characteristic curves ("ROC") above, our best PointNet model consisted of the following:

- All features: hits point cloud, momentum, and radius
- Time delta: 0.20ns
- Learning rate: Linear annealing scheduler.  Initial learning rate of 0.0003 that increases to 0.003 in the first half of training, before decreasing back to the original learning rate in the second half of training
- Epochs: 16 epochs of training

We selected an operating point of 0.93 on the ROC curve, as this allowed us to achieve a strong pion efficiency while maintaining good muon efficiency relative to the NA62 results.

### Dynamic Graph CNN

#### Model benefits

- Dynamic Graph CNN is designed to capture local information within the point cloud data
- Dynamic Graph CNN training time is almost two times faster than PointNet

#### Model shortcomings

- Although Dynamic Graph CNN can maintain a similar pion efficiency to PointNet, it stuggles with muon efficiency.

#### Best performing model

![dgcnn_roc](images/dgcnn_roc.png)

As can be seen in the receiver operating characteristic curves ("ROC") above, our best Dynamic Graph CNN model consisted of the following:

- All features: hits point cloud, momentum, and radius
- Time delta: 0.30ns
- Learning rate: Linear annealing scheduler.  Initial learning rate of 0.0003 that increases to 0.003 in the first half of training, before decreasing back to the original learning rate in the second half of training.
- Epochs: 8 epochs of training

We selected an operating point of 0.96 on the ROC curve, as this allowed us to achieve a strong pion efficiency.  We were unable to select an operating point that achieved both a good pion efficiency and a good muon efficiency.

## Overall model results

### Selecting the overall best model

![all_models_roc](images/all_models_roc.png)

In selecting our overall best model, we used the ROC curve.  As can be seen in the figure, our overall best model was PointNet, following by Dynamic Graph CNN.  Both deep learning models were able to surpass our baseline XGBoost model.
<!-- #endregion -->

### Model efficiencies

![dgcnn_efficiency](images/dgcnn_efficiency.png)
![pointnet_efficiency](images/pointnet_efficiency.png)

In analyzing our final model performance in terms of pion and muon efficiency, we note the following:

- Both the PointNet and Dynamic Graph CNN models were able to surpass the prior NA62 pion efficiency performance across all momentum bins.
- The Dynamic Graph CNN model was unable to achieve a similar muon efficiency to NA62 in any momentum bin.
- The PointNet model was able to surpass NA62 muon efficiency in momentum bins > 34 GeV/c.
