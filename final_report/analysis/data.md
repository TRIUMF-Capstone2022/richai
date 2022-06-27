# 2. Data 

## 2.1: Volume and Features

The data was generated as part of the NA62 experiments performed at CERN {cite}`https://doi.org/10.48550/arxiv.1706.08496`.  There are a total of 11 million labeled decay events, each containing features that can be utilized by machine learning models. However, there is a large class imbalance in the data set. Only 10% of the examples represent pion decays, the class of interest.

Based on the experimental setup detailed in {ref}`App. B.1 <appendix:datagen:process>`, two sets of features associated are derived for each event. The first set corresponds to the subatomic particle motion: the particle momentum and time spent in the detector ("CHOD" time). The second set of features are derived from the light emitted by the subatomic particle motion. Each photon detected by a PMT tube is recorded as a hit on the PMT grid with $x$ and $y$ coordinates relative to this grid and the time of the hit. As an abstraction to the hit scatter, the ring radius and center that result from the maximum likelihood fit that TRIUMF currently employs is also included.

## 2.3: Preprocessing

### 2.3.1: Subsampling w/ respect to momentum bins

The class imbalance will be detrimental to achieving a high pion efficiency as the pion is the minority class. Undersampling the muons to match the number of pions was a feasible solution to address this issue due to the large dataset. However, random sampling of muons examples cannot be used for this data. This is due to the systematic difference in the distribution of momentums between the two particles which is purely an artifact of the experimental setup as seen in {numref}`momentum_distribution`. This will bias the output as the objective of this project is to carry out the classification strictly based on differences in the ring size between the particles. 

```{figure} ../images/momentum_dist.svg
---
height: 700px
width: 950px
name: momentum_distribution
---

Distribution of momentum for all samples by class. 
```

The solution was to split the data into three equally sized bins by momentum in the range of 15-45 $GeV/c$, count the number of pions in each bin, and sample an equal number of muons within that bin. The resulting synthetic dataset contained 2 million examples. There were enough examples to feed into the machine learning models, and momentum as a feature was controlled.

### 2.3.2: Additional Steps

There were two additional preprocessing steps: removing the bias in the ring center and conversion of the hits data into a point cloud. The former was completed by subtracting the global mean $x$ and $y$ positions from every photon hit, and is shown in {ref}`App. B.2 <appendix:datagen:ringbias>`. The latter process, descibed in {ref}`App. B.3 <appendix:datagen:cloud>` involved adding a third dimension of time to the photon hit scatters.
