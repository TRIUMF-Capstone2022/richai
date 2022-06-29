# Configuration file

This directory contains the master configuration file which controls all of the hyperparameters for model training.  A summary of these hyperparameters is below:

## Data set hyperparameters

These are the hyperparameters related to the data set used for training.

| Hyperparameter                  | Description                                                                      | Value                                                 |
|---------------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------|
| `delta`                         | Time delta in ns between hit time and chod time to use for filtering.            | Float within (0, 1)                                   |
| `train`                         | Path to the training data (can contain multiple)                                 | Key:value where key = raw data, value = balanced data |
| `test`                          | Path to the testing data (can contain multiple)                                  | Same as above                                         |
| `dgcnn.val`                     | Validation portion for Dynamic Graph CNN training                                | Float within (0, 1)                                   |
| `dgcnn.test`                    | Testing portion for Dynamic Graph CNN training                                   | Float within (0, 1)                                   |
| `pointnet.val`                  | Validation portion for PointNet training                                         | Float within (0, 1)                                   |
| `pointnet.test`                 | Testing portion for PointNet CNN training                                        | Float within (0, 1)                                   |
| `centre_bias.mean_x`            | Global *x* coordinate for hit debiasing                                          | Float                                                 |
| `centre_bias.mean_y`            | Global *y* coordinate for hit debiasing                                          | Float                                                 |
| `standardize.mean_momentum`     | Global mean for momentum values to use for feature standardization               | Float                                                 |
| `standardize.std_momentum`      | Global standard deviation for momentum values to use for feature standardization | Float                                                 |
| `standardize.mean_radius`       | Global mean for radius values to use for feature standardization                 | Float                                                 |
| `standardize.std_radius`        | Global standard deviation for radius values to use for feature standardization   | Float                                                 |
| `filters.momentum_lower`        | Lower momentum bin for filtering events data                                     | Integer                                               |
| `filters.momentum_upper`        | Upper momentum bin for filtering events data                                     | Integer                                               |
| `filters.ring_center_pos_lower` | Lower bound for filtering outliers in ring centre location.                      | Integer                                               |
| `filters.ring_center_pos_lower` | Upper bound for filtering outliers in ring centre location.                      | Integer                                               |

## Data loader hyperparameters

These are the hyperparameters related to the data loaders used for training.

| Hyperparameter | Description                                                              | Value   |
|----------------|--------------------------------------------------------------------------|---------|
| `batch_size`   | The batch size to use for the data loaders                               | Integer |
| `num_workers`  | The number of CPU workers                                                | Integer |
| `drop_last`    | Whether or not to drop the last batch if it is not equal to `batch_size` | Boolean |

## Model hyperparameters

These are the hyperparameters related to the two models we trained, Pointnet and Dynamic Graph CNN.

### PointNet

| Hyperparameter    | Description                                                      | Value   |
|-------------------|------------------------------------------------------------------|---------|
| `output_channels` | The number of output classification channels in the network      | Integer |
| `epochs`          | The number of training epochs                                    | Integer |
| `saved_model`     | The path where to save the model to in `.pt` format              | String  |
| `predictions`     | The path where to save the test predictions in `.csv` format`    | String  |
| `train_result`    | The path where to save the training predictions in `.csv` format | String  |
| `momentum`        | Whether or not to include momentum as a feature in the network   | Boolean |
| `radius`          | Whether or not to include radius as a feature in the network     | Boolean |
| `seed`            | The seed to use for training                                     | Integer |
| `learning_rate`   | The learning rate to use for training                            | Float   |

### Dynamic Graph CNN

| Hyperparameter    | Description                                                      | Value   |
|-------------------|------------------------------------------------------------------|---------|
| `output_channels` | The number of output classification channels in the network      | Integer |
| `epochs`          | The number of training epochs                                    | Integer |
| `k`               | The value of `k` for the KNN graph                               | Integer |
| `saved_model`     | The path where to save the model to in `.pt` format              | String  |
| `predictions`     | The path where to save the test predictions in `.csv` format`    | String  |
| `train_result`    | The path where to save the training predictions in `.csv` format | String  |
| `momentum`        | Whether or not to include momentum as a feature in the network   | Boolean |
| `radius`          | Whether or not to include radius as a feature in the network     | Boolean |
| `seed`            | The seed to use for training                                     | Integer |
| `learning_rate`   | The learning rate to use for training                            | Float   |

## GPUs

Before training, list the desired GPUs within this hyperparameter like the following example (you can continue adding numbers as long as that GPU exists and has a corresponding CUDA id).:

```yaml
gpu:
    - 1
    - 2
    - 3
```

