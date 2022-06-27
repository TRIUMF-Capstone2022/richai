# Appendix C: Gradient Boosted Tree (ML Baseline)

(appendix:ml_gbm:intro)=
### C.1: Why are GBTs a good baseline model?

Gradient Boosted Decision Trees (GBDT) use ensemble of decision trees sequentially minimising a loss function and hence are popular due their efficiency, accuracy and ability to avoid overfitting. Besides, the libraries associated offer flexibility in terms of parameters such as decision tree algorithm, loss function, regularization, GPU related parameters etc. which make them a popular first choice as baseline models. 

There are several algorithm based GBDTs available on open source platforms such as Lightgbm, Catboost, Xgboost, Adaboost etc. They are mostly available in form of individual libraries with native implementations and with sklearn in some cases.
