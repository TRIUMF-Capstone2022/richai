# 5. Conclusions and recommendations

## 5.1: Conclusion

The goal of the RICH AI project was to build a deep learning classifier that was able to accurately classify a meson decay into either a pion or muon particle.  The main challenge of the project was to surpassing the prior performance of CERN's NA62 particle classification project, with a focus on the pion efficiency (pion recall) and muon efficiency (1 - muon recall).

As a solution, our final data product consisted of a machine learning pipeline that automates the model training, validation, testing, and prediction for raw input data.  Further, our final best model (PointNet), was able to surpass the prior CERN NA62 project in terms of pion efficiency, however, was unfortunately unable to achieve the same muon efficiency performance.  We note that the models may performance better with further tuning and/or additional training data.

## 5.2: Recommendations 

### 5.2.1: Removal of Bias in Ring Centers

As discussed in {ref}`App. B.3<appendix:datagen:ringbias>`, there exists a spatial bias in the ring centers which was inadequately addressed in this analysis. Further studies can seek to superimpose the mean ring centers of either class. Once further concern is that the PMT grid is in fact discreet (the photon hits can only land at certain coordinates), and therefore shifting the points using a mean can place these points in non-existent points. The effect of this needs to be studied. 

### 5.2.2: Operating points for each momentum bin 

The current comparison between the results of the PointNet and Dynamic Graph CNN model and the prior MLE estimate is based on the selection of a single operating point. The resulting pion and muon efficiencies for that operating point are compared with the MLE output across all bins. To further measure the performance of the DL models, the operating point can be selected to match the pion efficiency yeilded by the MLE algorithm at a particular momentum bin, and compare the muon efficiency. Though this requires extensive tuning, this is the most fair comparison between both models.


### 5.2.3: Further suggestions on models

We note that our final data product has room for improvement, and due to the time constraints of the capstone project, we did not implement everything that we would have liked to.  We leave the following suggestions to the TRIUMF team as potential future improvements.  The deep learning models could continue to be optimized, through a method such as Bayesian optimization.  Further, more muons could be added to the data set, while respecting the particle momentum bins, which may help the in generalizing better to muon decays and potentially improve muon efficiency.  Finally, once the models have been completely finalized and the scientists at TRIUMF decide that no further tuning is required, the models could be deployed.  Another more drastic suggestion would be to try other model architectures, such as PointNet++ {cite}`qi2017pointnet++`. 
