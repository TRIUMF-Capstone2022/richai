# 5. Conclusions and recommendations

The goal of the RICH AI project was to build a deep learning classifier that was able to accurately classify a meson decay into either a pion or muon particle.  The main challenge of the project was in surpassing the prior performance of CERN's NA62 partice classification project, as it relates to pion efficiency (pion recall) and muon efficiency (1 - mun recall).

We note that our final data product consisted of a machine learning pipeline that automates the model training, validation, testing, and prediction for raw input data.  Further, our final best model (PointNet), was able to surpass the prior CERN NA62 project in terms of pion efficiency, however, was unfortunately unable to achieve the same muon efficiency performance.  We note that the models may performance better with further tuning and/or additional training data.
