# Welcome!

## Overview

This [Jupyter Book](https://jupyterbook.org/en/stable/intro.html) contains the final report for the 2022 RICH AI Capstone Project completed by [Nico Van den Hooff](https://www.linkedin.com/in/nicovandenhooff/), [Rakesh Pandey](https://www.linkedin.com/in/rakeshpandey820/), [Mukund Iyer](https://www.linkedin.com/in/mukund-iyer19/), and [Shiva Jena](https://www.linkedin.com/in/shiva-jena/) as part of the University of British Columbia Master of Data Science program.  The project was completed in partnership with [TRIUMF](https://www.triumf.ca/), Canada's particle accelerator centre and one of the world's leading subatomic physics research centres.

## Executive summary

The NA62 experiment at CERN (European organization for nuclear research) studies the rate of the ultra-rate meson decay into a pion to verify the Standard Model in physics. The aim of this project is to develop a binary classification model based on advanced Machine Learning (ML) to distinguish pion decays from muon decays using the output of the Ring Imaging Cherenkov (RICH). The challenge lies in concurrently increasing the pion efficiency and mion efficiency and surpassing the performance of the MLE algorithm. The data used to build the machine learning models had 2 million examples, controlled for momentum and converted to point cloud form by the addition of a time dimension. Two deep learning models were applied: PointNet and Dynamic Graph CNN (DGCNN). Both were built using the point clouds of hits, particle momentum and the ring radius computed using the MLE. The best performing PointNet model used all these features, with a time delta of 0.2 ns and 16 epochs of training. Likewise the best performing DGCNN used all the features, k = 8 nearest neighbors and a  time delta of 0.3. The overall best performing model was PointNet as it has the highest AUC ROC, exceeds the pion efficiency from the MLE estimate for all momentum bins, and maintains a low muon efficiency for momentums beyond 34 GeV/c. Meanwhile, the DGCNN is able to maintain a similar pion efficiency but fails to maintain an adequate muon efficiency to surpass the MLE estimate. The final data product is a machine learning pipeline that takes in the experiment data in a HDP5 format, pre-processes it, prepares training data, trains classifier models on the training data and evaluates model results on test data through modular scripts.


## Table of contents

```{tableofcontents}
```
