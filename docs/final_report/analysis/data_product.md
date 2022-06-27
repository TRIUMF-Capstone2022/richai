# 4. Data Product

## Product overview

Our final data product is a modularized machine learning pipeline that ultimately can classify a particle decay with either PointNet or a Dynamic Graph CNN.  As input, the pipeline takes in raw experiement data in HDF5 file format.  The data is then preprocessed, and the models are trained and tested(alternatively, pre-trained models can also be loaded to skip the training process here).  Finally, the models can then be evaluated in order to classify a particle decay.  From a birds eye view, our pipeline works as shown in {numref}`product_overview`

```{figure} ../images/product_overview.png
---
name: product_overview
---

Machine learning pipeline
```

## How the product works

We have designed our data product to be designed by a master configuration file.  This configuration file is set before the scripts are run, and allows a user to set all of the desired parameters or hyperparameters related to a single experiment run.  At a high level, our modularized code has been structured into the following directories:

```
├── configs
├── src/
│   ├── dataset/
│   ├── models/
│   ├── train_evaluate/
│   └── utils/
```

This report will only include a very brief overview of each directory above, for full details please see the `README.md` file on our [GitHub repository](https://github.com/TRIUMF-Capstone2022/richai).

### `configs`

The configs directory cotains the master configuration file that allows the user to control all parameters and hyperparameters related to the data set, data loader, deep learning models, and selection of which GPUs to train on.  Please note that the master configuration file is the only file that has been designed to be edited by a user, and the changes in this file will flow down programatically to the python scripts that are described in the directories below.

### `dataset`

The dataset directory contains the source code that builds a custom PyTorch data set, as well as PyTorch data loaders that are used in model tranining, validation, and evaluation.

### `models`

The models directory contains the source code for our implementation of both PointNet and Dynamic Graph CNN.

### `train_evaluate`

The train_evaluate directory cotains the source code for our model training and evaluation scripts.

### `utils`

The utils directory contains other various functions related to our project, such as plotting functions or other functions that did not fit into one of the previous directories explained.

## Future improvements

We note that our final data product has room for improvement, and due to the time constraints of the capstone project, we did not implement everything that we would have liked to.  We leave the following suggestions to the TRIUMF team as potential future improvements.  The deep learning models could continue to be optimized, through a method such as Bayesian optimization.  Further, more muons could be added to the data set, while respecting the particle momentum bins, which may help the in generalizing better to muon decays and potentially improve muon efficiency.  Finally, once the models have been completely finalized and the scientists at TRIUMF decide that no further tuning is required, the models could be deployed.  Another more drastic suggestion would be to try other model architectures, such as PointNet++ {cite}`qi2017pointnet++`.
