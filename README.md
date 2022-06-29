# TRIUMF RICH AI

<!-- Badges start -->

[![final_report](https://github.com/TRIUMF-Capstone2022/richai/actions/workflows/final_report.yml/badge.svg)](https://github.com/TRIUMF-Capstone2022/richai/actions/workflows/final_report.yml) [![GitHub deployments](https://img.shields.io/github/deployments/TRIUMF-Capstone2022/richai/github-pages?label=gh-pages)](https://github.com/TRIUMF-Capstone2022/richai/deployments/activity_log?environment=github-pages) [![License](https://img.shields.io/github/license/TRIUMF-Capstone2022/richai)](https://github.com/TRIUMF-Capstone2022/richai/blob/main/LICENSE) [![GitHub release (latest by date)](https://img.shields.io/github/v/release/TRIUMF-Capstone2022/richai)](https://github.com/TRIUMF-Capstone2022/richai/releases)

<!-- Badges end -->

First and foremost, a big warm welcome! :balloon::tada: :confetti_ball: :balloon::balloon:

This GitHub repository contains the work related to the 2022 RICH AI Capstone Project completed by [Nico Van den Hooff](https://www.linkedin.com/in/nicovandenhooff/), [Rakesh Pandey](https://www.linkedin.com/in/rakeshpandey820/), [Mukund Iyer](https://www.linkedin.com/in/mukund-iyer19/), and [Shiva Jena](https://www.linkedin.com/in/shiva-jena/) as part of the University of British Columbia [Master of Data Science program](https://masterdatascience.ubc.ca/).  The project was completed in partnership with scientists at [TRIUMF](https://www.triumf.ca/), Canada's particle accelerator centre and one of the world's leading subatomic physics research centres.

This document (the `README.md` file) is a hub to give you some general information about the project. You can either navigate straight to one of the sections by using the links below, or simply scroll down to find out more.

- [TRIUMF RICH AI](#triumf-rich-ai)
  - [Executive summary](#executive-summary)
  - [Contributors](#contributors)
  - [Report](#report)
  - [Project structure](#project-structure)
  - [Dependencies](#dependencies)
  - [Configuration file](#configuration-file)
  - [Dataset](#dataset)
  - [Model training](#model-training)
  - [Model evaluation and scoring on new data](#model-evaluation-and-scoring-on-new-data)
  - [Utility scripts](#utility-scripts)
  - [Jupyter notebooks](#jupyter-notebooks)
  - [Code of conduct](#code-of-conduct)
  - [Contributing](#contributing)
  - [License](#license)
  - [References](#references)

## Executive summary

The [NA62 experiment](https://home.cern/science/experiments/na62) at [CERN](https://home.cern/) (European organization for nuclear research) studies the rate of the ultra-rare meson decay into a pion particle to verify the [Standard Model](https://en.wikipedia.org/wiki/Standard_Model) in physics. The aim of the RICH AI project is to develop a binary classification model that utilizes advanced Machine Learning ("ML") to distinguish pion decays from muon decays by using the output of a [Ring Imaging Cherenkov](https://en.wikipedia.org/wiki/Ring-imaging_Cherenkov_detector) ("RICH") detector. The challenge of the project lies in concurrently *increasing* the "pion efficiency" (the rate at which pion decays are correctly classified) while also *decreasing* the muon efficiency (the rate at which muon decays are incorrectly classified as pion decays), in order to surpass the performance of CERN's current algorithm that is employed (i.e. a simple maximum likelihood algorithm that fits a circle to the light image emitted by a particle decay, and classifies the particle based on an expected radius size).  

The data used to build the machine learning models had 2 million examples. It was controlled for momentum and converted to point cloud form by the addition of a time dimension. Two deep learning models based on recent academic research were applied: [PointNet](https://arxiv.org/abs/1612.00593) and [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829) ("DGCNN"). The overall best performing model was PointNet as it exceeded the prior pion efficiency achieved by CERN's NA62 algorithm, while also maintaining a similar muon efficiency. 

The final data product of the project consists of a modularized machine learning pipeline that takes in the raw experiment data in a HDF5 format, pre-processes it, prepares training data, trains a classifier model on the training data (or alternatively loads a trained model), and finally evaluates the model to make a prediction. In further studies, there is room for more work to be done on debiasing the ring centre locations of the particles, as well as further hyperparameter tuning for the machine learning models.

## Contributors

- [Nico Van den Hooff](https://www.linkedin.com/in/nicovandenhooff/)
- [Rakesh Pandey](https://www.linkedin.com/in/rakeshpandey820/)
- [Mukund Iyer](https://www.linkedin.com/in/mukund-iyer19/)
- [Shiva Jena](https://www.linkedin.com/in/shiva-jena/)

## Report

The final report of the project, which contains much greater detail about the data and modelling processes can be accessed [here](https://triumf-capstone2022.github.io/richai/welcome.html).

### Jupyter book

The report is hosted as a [Jupyter Book](https://jupyterbook.org/en/stable/intro.html) on GitHub pages, and the underlying files that are used to build the Jupyter Book can be accessed [here](https://github.com/TRIUMF-Capstone2022/richai/tree/main/docs/final_report).  The Jupyter Book itself is built automatically via a [GitHub Actions workflow](https://github.com/TRIUMF-Capstone2022/richai/blob/main/.github/workflows/final_report.yml), which triggers if there is a push to the `main` branch of this repository that changes a file within the `richai/docs/final_report/` directory.

### Final presentation slides

The corresponding slides for the final presentation that was given to the UBC Master of Data Science faculty and cohort can be accessed [here](https://github.com/TRIUMF-Capstone2022/richai/tree/main/docs/final_presentation).

### Project proposal

Finally, the original project proposal and corresponding presentation slides can be accessed [here](https://github.com/TRIUMF-Capstone2022/richai/tree/main/docs/proposal).

## Project structure

At a high level, the overall project structure is as follows:

```
.
├── configs/
│   ├── README.md
│   └── config.yaml
├── dataset/
│   ├── balance_data.py
│   ├── data_loader.py
│   ├── rich_dataset.py
│   └── rich_pmt_positions.npy
├── docs/
│   ├── final_presentation/
│   ├── final_report/
│   ├── images/
│   ├── proposal/
│   └── README.md
├── models/
│   ├── dgcnn.py
│   └── pointnet.py
├── notebooks/
│   ├── DGCNN_operating_point.ipynb
│   ├── EDA.ipynb
│   ├── README.md
│   ├── balance_data.ipynb
│   ├── data_generation_process.ipynb
│   ├── gbdt_analysis_results.ipynb
│   ├── global_values.ipynb
│   ├── plotting_NA62_rings.ipynb
│   ├── pointnet_model_runs.ipynb
│   ├── pointnet_operating_point.ipynb
│   └── presentation_plots.ipynb
├── saved_models/
│   └── README.md
├── src/
│   ├── evaluate.py
│   └── train.py
├── utils/
│   ├── gbt_dataset.py
│   ├── helpers.py
│   └── plotting.py
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
└── README.me
```

## Dependencies

The RICH AI project was developed using [`singularity`](https://docs.sylabs.io/guides/3.0/user-guide/index.html) containers with the following package dependencies.

- pandas==1.3.5
- torch==1.11.0
- sklearn==0.24.0
- pyyaml==6.0
- jupyterlab

## Configuration file

The configuration file contains all of the parameters for the dataset, filters, model training, and scoring.
File [`configs/config.yaml`](https://github.com/TRIUMF-Capstone2022/richai/tree/main/configs) can be used to control data set paths, filters, model parameters/hyperparameters, train/test/saved model paths, PyTorch `Dataloader` configurations such as batch size, number of workers, etc., number of training epochs, device id, and many more.

> Before beginning the training process, it is recommended that you double-check that the configuration parameters, such as datset and model paths, are correct.

## Dataset

The data was generated as part of the 2018 [NA62 experiments](https://home.cern/science/experiments/na62) performed at CERN. There are a total of 11 million labeled decay events, each containing the features detailed above. However, there was a large class imbalance in the data set as only 10% of the decay examples were pion decays (the class of interest). More details can be [here](https://triumf-capstone2022.github.io/richai/analysis/data.html).

The sub directory [`dataset`](https://github.com/TRIUMF-Capstone2022/richai/tree/main/dataset) contains scripts for creating a custom PyTorch `Dataset` and `DataLoader` for deep learning models, along with a `balance_data` to create balanced dataset by undersampling the higher sized class.

- [`rich_dataset.py`](https://github.com/TRIUMF-Capstone2022/richai/blob/main/dataset/rich_dataset.py) processes the raw project data from HDF5 format and extracts events, hits and position data into a custom PyTorch `Dataset`.
- [`dataloader.py`](https://github.com/TRIUMF-Capstone2022/richai/blob/main/dataset/data_loader.py) creates PyTorch `DataLoader`s used to load data (train/test/validation) in batches as feed into the neural network models.
- [`balance_data.py`](https://github.com/TRIUMF-Capstone2022/richai/blob/main/dataset/balance_data.py) reads HDF5 files from the provided source file paths, creates balanced data by undersampling the higher sized class, and saves the HDF5 file to the specified path. Usage details can be found in the notebooks.

The data set configuration can be controlled and customized using the `dataset` section of [configuration file](#configuration-file).

## Model training

To train `PointNet` use the following command at the root directory.

```bash
python src/train.py --model pointnet
```

To train `Dynamic Graph CNN` use the following command at the root directory.

```bash
python src/train.py --model dgcnn
```

> The trained model object can be found at the path specified in `configs/config.yaml` as `model.<model_name>.saved_model`.

## Model evaluation and scoring on new data

To evaluate `PointNet` on test data or to score on a new data, use the following command at the root directory.

```bash
python src/evaluate.py --model pointnet
```

To evaluate `Dynamic Graph CNN` on test data or to score on a new data, use the following command at the root directory.

```bash
python src/evaluate.py --model dgcnn
```

> Model scored `.csv` data can be found in the path specified in `configs/config.yaml` as `model.<model_name>.predictions`. It contains actual labels, predicted labels, and predicted probabilities.

> Saved models trained with different configurations and corresponding results as `.csv` files can be found on the `triumf-ml1` server at the path `/fast_scratch_1/capstone_2022/models/`. Please refer [appendix in the final report](https://triumf-capstone2022.github.io/richai/appendix/supplementary_notebooks.html) to learn more about the different model runs. 

## Utility scripts

The `richai/utils` directory contains the following scripts:

- `helpers.py` which contains code for useful helper functions that were used throughout the project.
- `plotting.py` which contains code for useful plotting functions that were used throughout the project.
- `gbt_dataset.py` which contains the code for the data set used for the XGBoost model

## Jupyter Notebooks

A number of Jupyter Notebooks were written during the course of the project to support the project analysis and to perform procedures such as Exploratory Data Analysis.  The actual Jupyter Notebooks are saved [here](https://github.com/TRIUMF-Capstone2022/richai/tree/main/notebooks).  Alternatively, they are also included in the final project report within the [supplementary Jupyter Notebooks section](https://triumf-capstone2022.github.io/richai/appendix/supplementary_notebooks.html) of the report appendix (it is easier to view them here, rather than on the GitHub website).

## Code of conduct

The project code of conduct can be found [here](https://github.com/TRIUMF-Capstone2022/richai/blob/main/CODE_OF_CONDUCT.md).

## Contributing

The project contributing file can be found [here](https://github.com/TRIUMF-Capstone2022/richai/blob/main/CONTRIBUTING.md).

## License

All work for the RICH AI capstone project is performed under an MIT license, which can be found [here](https://github.com/TRIUMF-Capstone2022/richai/blob/main/LICENSE).

## References

For a full list of project references, please see the [references](https://triumf-capstone2022.github.io/richai/references.html) section of the final report.
