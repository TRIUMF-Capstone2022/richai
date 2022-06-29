(appendix:deeplearning:pointnet)=
# Appendix D: PointNet

(appendix:deeplearning:pointnet:arch)=
## D.1: Model architecture 

### D.1.1: Base Architecture

The PointNet architecture has two key features which allow it to handle pointcloud data effectively: 

1. Spatial transformation network (T-Net) {cite}`https://doi.org/10.48550/arxiv.1506.02025` 
2. Maxpooling

The overall architecture can be seen in {numref}`pointnetarch` {cite}`qi2017pointnet`. The spatial transformer ("T-net") is the first part of the model, and makes the model invariant to geometric transformations in the point cloud. The parameters for the matrix learnt by this network are data driven and therefore robust. The second application of T-Net acts on the features extracted from various point clouds and orient them in a latent feature space. This matrix learnt for this step is more complex. 

These geometric invariant features are passed onto a fully connected layer for feature extraction. Subsequently, max pooling, a symmetric function, is applied to this output to capture the unordered nature of the points in the point cloud. The output is passed to a classification network. 


```{figure} ../images/pointnetarch.png
---
height: 300px
width: 900px
name: pointnetarch
---

The architecture of the base PointNet model. 
```

### D.1.2: Additional features 

The particle momentum and ring radius were added to this architecture by concatenating them to the fully connected layer. Both features were standardized and normalized batchwise to ensure stability of the network. 

(appendix:deeplearning:pointnet:hyp)=
## D.2: Model hyperparamters 

The hyperparameters that were tuned were:

- Time delta between the hit time and the CHOD time, for which values of 0.20ns to 0.50ns were used
- Learning rate, for which a constant learning rate and a learning rate scheduler were used
- Number of epochs, for which a maximum of 24 epochs were used
