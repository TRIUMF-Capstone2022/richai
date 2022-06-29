# 4. Data Product

## 4.1: Product overview

The final data product of RICH AI is a modularized machine learning pipeline that can classify a particle decay with either PointNet or Dynamic Graph CNN.  As input, the pipeline takes in raw experiment data in HDF5 file format.  The data is then preprocessed, and the models are trained and tested(alternatively, pre-trained models can also be loaded to skip the training process here).  Finally, the models can then be evaluated in order to classify a particle decay.  From a birds eye view, the pipeline works as shown in {numref}`product_overview`

```{figure} ../images/product_overview.png
---
name: product_overview
---

Machine learning pipeline
```

## 4.2: How the product works

The product as been designed to be controlled by a master configuration file.  This configuration file is set before the scripts are run, and this allows a user to set all of the desired hyperparameters related to a single experiment run.  At a high level, the modularized code has been structured into the following directories:

```
├── configs/
├── dataset/
├── models/
├── src/
├── utils/
```

This report will only include a very brief overview of each directory above, for full details please see the `README.md` file of the project's [GitHub repository](https://github.com/TRIUMF-Capstone2022/richai).

### 4.2.1: `configs`

The `configs` directory contains the master configuration file that allows the user to control all parameters and hyperparameters related to the data set, data loader, deep learning models, and selection of which GPUs to train on.  Please note that the master configuration file is the only file that has been designed to be edited by a user, and the changes in this file will flow down programmatically to the python scripts that are described in the directories below.

### 4.2.2: `dataset`

The `dataset` directory contains the source code that builds a custom PyTorch `Dataset`, as well as PyTorch `DataLoader`s that are used in model training, validation, and evaluation.

### 4.2.3: `models`

The `models` directory contains the source code for the implementation of both PointNet and Dynamic Graph CNN.

### 4.2.4: `src`

The `src` directory contains the source code for the model training and evaluation scripts that can be used for either deep learning model.

### 4.2.5: `utils`

The `utils` directory contains other various functions related to the project, such as plotting functions or other functions that did not fit into one of the previous directories explained.

