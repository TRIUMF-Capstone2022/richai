# 5. Conclusions and recommendations

## 5.1: Conclusion

The goal of the RICH AI project was to build a binary machine learning classifier that would be able to accurately classify a meson decay into either a pion or muon particle.  The main challenge of the project was in surpassing the prior performance of CERN's NA62 particle classification project, with a focus on the pion efficiency and muon efficiency.

As a solution, the final data product created consisted of a machine learning pipeline that automates the model training, validation, testing, and prediction for raw input data.  Further, the best model (PointNet), was able to surpass the prior CERN NA62 project in terms of pion efficiency, however, was unfortunately unable to achieve the same muon efficiency performance. The model may perform better with further tuning and additional training data.

## 5.2: Recommendations 

### 5.2.1: Removal of Bias in Ring Centers

As discussed in {ref}`App. B.3<appendix:datagen:ringbias>`, there exists a spatial bias in the location of the ring centres determined by CERN's prior algorithm, which was unaddressed in this analysis. Further studies can seek to superimpose the mean ring centers of either class. Once further concern is that the PMT grid is in fact discrete (the photon hits can only land at certain coordinates), and therefore shifting the points using can place these points in non-existent PMT locations. The effect of this needs to be further studied. 

### 5.2.2: Operating points for each momentum bin 

The current comparison between the results of the PointNet and Dynamic Graph CNN model and the CERN algorithm is based on the selection of a single operating point for both models. The resulting pion and muon efficiencies for that operating point are compared with the CERN algorithm across all bins. To further measure the performance of the deep learning models, an operating point could be selected to match the pion efficiency yielded by the CERN algorithm at individual momentum bins, and then the muon efficiencies could be compared.  This could be explored in future studies.


### 5.2.3: Further suggestions for deep learning

The deep learning models could continue to be optimized, through a method such as Bayesian optimization.  Further, more muons could be added to the data set, while respecting the particle momentum bins, which may help the in generalizing better to muon decays and potentially improve muon efficiency.  Finally, once the models have been completely finalized the models could be productionized and deployed.  Another more drastic suggestion would be to try other model architectures, such as PointNet++ {cite}`qi2017pointnet++`. 
