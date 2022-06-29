# Appendix B: Data 

(appendix:datagen:process)=
## B.1: Data Generation Process ###

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

4. The cone of light is reflected by a mosaic of mirrors onto an array of photomultiplier tubes (“PMT”).  In an ideal situation, the cone of light forms a “ring” on the PMT array.

5. Each individual PMT in the array records whether or not it was hit by light, as well as the time of arrival for each hit of light.

6. The hodoscope counters (CHOD) detector shown in Figure 2 records the time that the particle decay occurs.

## B.2: Bias in Data

(appendix:datagen:ringbias)=
### Ring center

```{figure} ../images/ring_cent_bias.png
---
height: 450px
width: 500px
name: ring_center
---

Distribution of ring center calculated using the MLE algorithm for all samples by class. 
```

Similarly,  {numref}`ring_center` details the difference in the ring centers, computed using the MLE, between the two classes. As this feature is an abstraction of the scatter of X and Y coordinates, it can be inferred that the bias exists in the raw hits information which will be fed into the deep learning models. These models can identify this spatial difference on the standardized PMT grid and bias the classification. Though the global X and Y coordinates of the mean ring center (over both classes) was subtracted from each photon hit coordinate in the analysis, this method is likely inadequate as it is simply a translation. This factor needs to be investigated further in subsequent studies. 

(appendix:datagen:cloud)=
## B.3: Point Cloud Generation

```{figure} ../images/eda_ring_plot.svg
---
name: ring_plot
---

Origial 2D structure of scatter of hits.
```

Naturally, the photon hits information for each event is a system of X and Y coordinates that exists in a 2D plane {numref}`ring_plot`. As the number of hits for an event usually ranges from 15-30, and the PMT grid is of size 1952, the information is very sparse when treated as a standard image. Furthermore, implementations of PointNet {ref}`App D.1 <appendix:deeplearning:pointnet:arch>` and the Dynamic Graph CNN{ref}`App E.1 <appendix:deeplearning:dgcnn:arch>` require the feed to be three dimensional. Hence the photon hits information was converted to a point cloud by adding a third dimension of time. Specifically, this was the absolute value of the difference between the photon hit time and the particle travel. This is an eloquent solution as noise hits will have a large value in this new dimension and therefore be separated from the ring produced by the genuine motion of the particle.

(appendix:datagen:delta)=
## B.4: Delta to filter noise hits

```{figure} ../images/photon_hits.svg
---
name: photon_hits
---

Time difference between the photon hit time and CHOD time for 1 event. The CHOD time is the moment at which the particle travels through the RICH detector and the photons of light are emitted.
```

The photon hits recorded for a particular event are noisy due to the limited instrument precision {numref}`photon_hits`. For an event, the data for photon hits may contain hits from a previous or subsequent event in addition to those from the current event {numref}. To correctly filter these, the hit times need to match closely with the CHOD time, which is the measure of the time at which the particle passes the RICH detector. As the photons are emitted in this exact process, the time difference between the two should be zero (or very close to that value). Therefore by filtering the hits using a thredhold, the hits information for every event follows a gausian distribution centered at zero, indicating that the hits only have statistical variance {numref}`delta_time`. This filtering parameter is referred to as *delta* in this study.

```{figure} ../images/eda_delta_time.svg
---
name: delta_time
---

Distribution of the difference between the photon hit times and CHOD time for a specific event.
```
