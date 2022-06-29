# 2. Data 

## 2.1: Volume and Features

The data set was generated as part of the NA62 experiments performed at CERN {cite}`https://doi.org/10.48550/arxiv.1706.08496`.  In total, there is approximately 11 million labeled particle decay events.  Each labeled event contains various features that can be utilized by machine learning models in training. However, a large class imbalance exists in the dataset, as only 10% of the total examples (approximately 1 million examples) represent pion decay events, which are the decay class of interest.

Based on the experimental setup detailed in {ref}`App. B.1 <appendix:datagen:process>`, there are two sets of features that are associated for each event. The first set corresponds to the subatomic particle motion: the *particle momentum* and *time* spent in the detector ("CHOD" time). The second set of features are derived from the light that is emitted by the subatomic particle motion during a decay. Each light photon detected by a photomultiplier tube ("PMT") tube is recorded as a "hit" on a grid of PMTs with *x* and *y* coordinates relative to this grid, along with the *time* of the hit. Also available as features are the ring radius and ring centre location that are calculated using CERN's prior algorithm that is used to currently classify the particle decays {cite}`na622020investigation`.

## 2.3: Preprocessing

### 2.3.1: Subsampling w/ respect to momentum bins

Using the raw data set as a training set would be detrimental to our overall goals due to the class imbalance that exists.  In other words, it would be difficult to obtain a high pion efficiency with the raw data, due to the lack of examples of pion decay events.  In order to deal with this imbalance, undersampling was employed.  Specifically, undersampling the number of muons to match the number of pions was performed. 

However, it is important to note that for this project, undersampling of muons cannot be performed naively. This is due to a systematic difference that exists in the distributions of particle momentums for muons and pions as seen in {numref}`momentum_distribution`.  This difference is an artifact of the experimental setup of the project.  In simpler terms, when undersampling the number of muons, the goal is the match the number of muons to the number of pions based on the number of pions in a *specific bin of particle momentums*. Otherwise, a machine learning model may learn a separation in the particle momentum distributions as an important feature in prediction, which in reality will not generalize to new data.

```{figure} ../images/eda_p_dist.svg
---
height: 405px
width: 550px
name: momentum_distribution
---

Distribution of momentum for all samples by class. 
```

In order to undersample, the data was split into three equally sized bins by particle momentum in the range of 15-45 *GeV/c*.  Then, the total number of pions in each bin was calculated, and an equal number of muons with the same particle momentum was randomly sampled to match the number of pions. The resulting synthetic balanced dataset contained a total of 2 million examples.

### 2.3.2: Additional Bias in ring center locations 

There exists a bias in the ring center locations for either particle as an artifact of the experiment, which is discussed in {ref}`App. B.2 <appendix:datagen:ringbias>`.

### 2.3.3: Conversion to point cloud data 

Before the PMT light hit coordinates were utilized as features in the deep learning models, they were converted into a point cloud of *(x, y, z)* coordinates by adding a third dimension of time.  Specifically, this third dimension of time was the time difference ("delta") between the time the light hit the PMT detector, and the time that the particle entered the RICH detector ("CHOD" time). The process is detailed further in {ref}`App. B.3 <appendix:datagen:cloud>`. The appendix also details why the conversion was necessary.

### 2.3.4: Filtering noise hits 

Due to the sensitivity of the detectors, the data recorded for photon hits corresponding to each event was noisy and contained hits from a prior or subsequent event(s). Therefore further data cleaning was performed as discussed in {ref}`App. B.4 <appendix:datagen:delta>`. 

## Summary of features

Ultimately, the following features were available for our machine learning models:

- Particle momentum
- Light hit locations on PMT tubes (x, y)
- Time of each hit on a PMT tube
- Time particle entered RICH detector (CHOD time)
- Fitted ring radius (mm) from CERN's prior algorithm
- Ring centre location from CERN's prior algorithm (not used in our models)

The point cloud data was created as:

- Light hit locations on PMT tubes (x, y) with the addition of (time of hit - CHOD time) as a z dimension to create a point cloud of *(x, y, z)* coordinates.


