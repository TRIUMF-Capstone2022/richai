# Appendix 2.1 : Data 

(appendix:datagen:process)=
## 2.1.1 Data Generation Process ###

In order to generate the NA62 data, several experiment "runs" are performed.  For each run, the experiment configuration is fixed and the following steps are performed:

```{figure} ../images/NA62_beam_detector.png
---
height: 250px
width: 1000px
name: NA62_beam_detector
---

Cross section of the setup of the NA62 experiment
```

1. A beam rich in kaon particles is delivered in “bursts” every four or five seconds into the detector. The set up as shown in {numref}`NA62_beam_detector` {cite}`Gil_2017`.

2. During a burst, several particle decays occur.  Each particle decay has an individual “event” ID associated with it.

```{figure} ../images/RICH_detector.png
---
height: 250px
width: 525px
name: rich_detector
---

Ring-imaging Cherenkov detector
```
3. The product of the decay is accelerated through a chamber of neon gas in the RICH detector and a cone of light is emitted.  The RICH detector is shown in {numref}`rich_detector` {cite}`anzivino2020light`.


### Debiasing w/ respect to ring center bias

```{figure} ../images/ring_cent_bias.png
---
height: 450px
width: 500px
name: ring_center
---

Distribution of ring center calcualted using the MLE algorithm for all samples by class. 
```

Similarly,  {numref}`ring_center` details the difference in the ring centers computed using the MLE between the two classes. As this feature is an abstraction of the scatter of X and Y coordinates, it can be inferred that the bias exists in the raw hits information which will be fed into the deep learning models. These models identify this spatial difference on the standardized PMT grid, and therefore bias the classification. Demeaning each of the hits data using the global X and Y positions of the ring centers, irrespective of class, will remove this bias. 

### Point cloud generation

```{figure} ../images/point_cloud.png
---
height: 350px
width: 1250px
name: point_cloud
---

The generation of the point cloud from the 2D scatter through the addtion of a time dimension. 
```
Naturally, the photon hits information for each event is a system of X and Y coordinates that exists in a 2D plane. As the number of hits for an event usually ranges from 15-30, and the PMT grid is of size 1952, the information is very sparse when treated as a standard image. Hence the photon hits information was converted to a point cloud by adding a third dimension of time: the absolute value of the difference between the photon hit time and the particle travel. This is an eloquent solution as noise hits will have a large value in this new dimension and therefore be separated from the ring produced by the genuine motion of the particle. This is detailed in {numref}`point_cloud`.

4. The cone of light is reflected by a mosaic of mirrors onto an array of photomultiplier tubes (“PMT”).  In an ideal situation, the cone of light forms a “ring” on the PMT array.
5. Each individual PMT in the array records whether or not it was hit by light, as well as the time of arrival for each hit of light.
6. The hodoscope counters (CHOD) detector shown in Figure 2 records the time that the particle decay occurs.


