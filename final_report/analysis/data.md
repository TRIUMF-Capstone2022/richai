# 2. Data 

## Features

Based on the experimental setup detailed in {ref}`App. B.1 <appendix:datagen:process>`, two sets of features associated are derived for each event. The first set corresponds to the subatomic particle motion: the particle momentum and time spent in the detector (CHOD time). The second set of features are derived from the light emitted by the subatomic particle motion. Each photon detected by a PMT tube is recorded as a hit on the PMT grid with X and Y coordinates relative to this grid and the time of the hit. As an abstraction to the hit scatter, the ring radius and center that result from the MLE fit that TRIUMF currently employ are also included. All in all, there are a total of 5 features for each event. 

## Volume

The data was generated as part of the 2018 NA62 experiments performed at CERN.  There are a total of 11 million labeled decay events, each containing the features detailed above. However, there is a large class imbalance in the data set. Only 10% of the examples are of pions, the class of interest.  

## Preprocessing

### Subsampling w/ respect to momentum bins ###

The class imbalance will be detrimental to achieving a high pion efficiency as the pion is the minority class. Undersampling the muons to match the number of pions was a feasible solution to address this issue due to the large dataset. However, random sampling of muons examples cannot be used for this data. This is due to the systematic difference in the distribution of momentums between the two particles and is purely an artifact of the experimental setup as seen in {numref}`momentum_distribution`. This will bias the output as the objective of this project is to carry out the classification strictly based on differences in the ring size between the particles. 

```{figure} ../images/momentum_distribution.pdf
---
height: 500px
width: 750px
name: momentum_distribution
---

Distribution of momentum for all samples by class. 
```

The solution was to split the data into three equally sized bins by momentum in the range of 15-45 $GeV/c$, count the number of pions in each bin, and sample an equal number of muons within that bin. The resulting synthetic dataset contained 2 million examples. There were enough examples to feed into the machine learning models, and momentum as a feature was controlled. 

### Additional Steps

There were two additional preprocessing steps: removing the bias in the ring center and conversion of the hits data into a point cloud. The former was completed by subtracting the global mean X and Y positions from every photon hit, and is shown in {ref}`App. B.2 <appendix:datagen:ringbias>`. The latter process, descibed in {ref}`App. B.3 <appendix:datagen:cloud>` involved adding a third dimension of time to the photon hit scatters. 
