# TRIUMF RICH AI

First and foremost, a big warm welcome! :balloon::tada: :confetti_ball: :balloon::balloon:

The 2022 RICH AI Capstone Project is completed at the University of British Columbia Master of Data Science programme in collaboration with [TRIUMF](https://www.triumf.ca/), Canada's particle accelerator centre and one of the world's leading subatomic physics research institutions.

This document (the README file) is a hub to give you some information about the project. You can either get straight to one of the sections below, or simply scroll down to find out more.

- [TRIUMF RICH AI](#triumf-rich-ai)
  - [About this project](#about-this-project)
  - [Contributors](#contributors)
  - [Report](#report)
  - [Project Structure](#project-structure)
  - [Model Training](#model-training)
  - [Model Evaluation and Scoring on new data](#model-evaluation-and-scoring-on-new-data)
  - [Configuration file](#configuration-file)

## About this project

The NA62 experiment at CERN (European organization for nuclear research) studies the rate of the ultra-rate meson decay into a pion to verify the Standard Model in physics. The aim of this project is to develop a binary classification model based on advanced Machine Learning (ML) to distinguish pion decays from muon decays using the output of the Ring Imaging Cherenkov (RICH). The challenge lies in concurrently increasing the pion efficiency and mion efficiency and surpassing the performance of the MLE algorithm.

The data used to build the machine learning models had 2 million examples, controlled for momentum and converted to point cloud form by the addition of a time dimension. Two deep learning models were applied: PointNet and Dynamic Graph CNN (DGCNN). Both were built using the point clouds of hits, particle momentum and the ring radius computed using the MLE. The best performing PointNet model used all these features, with a time delta of 0.2 ns and 16 epochs of training. Likewise the best performing DGCNN used all the features, k = 8 nearest neighbors and a  time delta of 0.3. The overall best performing model was PointNet as it has the highest AUC ROC, exceeds the pion efficiency from the MLE estimate for all momentum bins, and maintains a low muon efficiency for momentums beyond 34 GeV/c. Meanwhile, the DGCNN is able to maintain a similar pion efficiency but fails to maintain an adequate muon efficiency to surpass the MLE estimate.

## Contributors

- [Nico Van den Hooff](https://www.linkedin.com/in/nicovandenhooff/)
- [Rakesh Pandey](https://www.linkedin.com/in/rakeshpandey820/)
- [Mukund Iyer](https://www.linkedin.com/in/mukund-iyer19/)
- [Shiva Jena](https://www.linkedin.com/in/shiva-jena/)

## Report

The final report can be accessed [here](https://github.com/TRIUMF-Capstone2022/richai/jupyter-book/final_report/).

## Project Structure

![Project Structure](docs/images/project_org.png)

## Model Training

To train PointNet use the following command.

```bash
python src/train.py --model pointnet
```

To train `Dynamic Graph CNN` use the following command.

```bash
python src/train.py --model dgcnn
```

## Model Evaluation and Scoring on new data

To evaluate `PointNet` on test data or to score on a new data, use the following command.

```bash
python src/evaluate.py --model pointnet
```

To evaluate `Dynamic Graph CNN` on test data or to score on a new data, use the following command.

```bash
python src/evaluate.py --model dgcnn
```

## Configuration file

All the parameters related to dataset, filters, model training and scoring are defined in the configuration file.
