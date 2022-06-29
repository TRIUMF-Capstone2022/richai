# Welcome!

## Overview

This [Jupyter Book](https://jupyterbook.org/en/stable/intro.html) contains the final report for the 2022 RICH AI Capstone Project completed by [Nico Van den Hooff](https://www.linkedin.com/in/nicovandenhooff/), [Rakesh Pandey](https://www.linkedin.com/in/rakeshpandey820/), [Mukund Iyer](https://www.linkedin.com/in/mukund-iyer19/), and [Shiva Jena](https://www.linkedin.com/in/shiva-jena/) as part of the University of British Columbia [Master of Data Science program](https://masterdatascience.ubc.ca/).  The project was completed in partnership with scientists at [TRIUMF](https://www.triumf.ca/), Canada's particle accelerator centre and one of the world's leading subatomic physics research centres.

## Executive summary

The [NA62 experiment](https://home.cern/science/experiments/na62) at [CERN](https://home.cern/) (European organization for nuclear research) studies the rate of the ultra-rare meson decay into a pion particle to verify the [Standard Model](https://en.wikipedia.org/wiki/Standard_Model) in physics. The aim of the RICH AI project is to develop a binary classification model that utilizes advanced Machine Learning ("ML") to distinguish pion decays from muon decays by using the output of a [Ring Imaging Cherenkov](https://en.wikipedia.org/wiki/Ring-imaging_Cherenkov_detector) ("RICH") detector. The challenge of the project lies in concurrently *increasing* the "pion efficiency" (the rate at which pion decays are correctly classified) while also *decreasing* the muon efficiency (the rate at which muon decays are incorrectly classified as pion decays), in order to surpass the performance of CERN's current algorithm that is employed (i.e. a simple maximum likelihood algorithm that fits a circle to the light image emitted by a particle decay, and classifies the particle based on an expected radius size).  

The data used to build the machine learning models had 2 million examples. It was controlled for momentum and converted to point cloud form by the addition of a time dimension. Two deep learning models based on recent academic research were applied: [PointNet](https://arxiv.org/abs/1612.00593) and [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829) ("DGCNN"). The overall best performing model was PointNet as it exceeded the prior pion efficiency achieved by CERN's NA62 algorithm, while also maintaining a similar muon efficiency. 

The final data product of the project consists of a modularized machine learning pipeline that takes in the raw experiment data in a HDF5 format, pre-processes it, prepares training data, trains a classifier model on the training data (or alternatively loads a trained model), and finally evaluates the model to make a prediction. In further studies, there is room for more work to be done on debiasing the ring centre locations of the particles, as well as further hyperparameter tuning for the machine learning models.

## Table of contents

```{tableofcontents}
```
