#!/usr/bin/env python
# coding: utf-8

# # RICH AI Data Generation Process

# **Purpose**: The purpose of this notebook is to demonstrate the data generation process for a single particle decay event as recorded by the RICH detector.
# **Author**: Nico Van den Hooff

# In[1]:


import h5py 
import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


# # Part 1: Reading in the data
# 
# - Currently, we are working with the 2018 dataset, which is split into three folders: `A`, `B`, and `C`.
# - The data is saved in `HDF5` format, and each subset of data contains:
# 
# | Item             | Explanation                                                                   |
# |------------------|-------------------------------------------------------------------------------|
# | **Events**       | Event data on each particle decay event.                                      |
# | **Hits**         | Hits data on each particle decay event.                                       |
# | **Hits mapping** | Used to determine how to index the hits array to find the hits for one event. |

# In[2]:


# data folder, data set, individual data file
data_folder = "/data/bvelghe/capstone2022/"
data_set = "A/"
file_path = "Run008548.EOSlist.CTRL.p.v2.0.4-01_f.v2.0.4-01.h5"

# read in 
f = h5py.File(os.path.join(data_folder, data_set, file_path))
f.keys()


# In[3]:


events=f['Events']        # event data
hits=f['Hits']            # pm hit data for each hit
hit_map=f['HitMapping']   # maps range of hits for each event


# ## 1.1 Event data
# 
# - This section shows how to extract the features from the `Events` data in an `HDF5` file.
# - We use event `0` here and throughout the notebook for demonstration purposes.

# In[4]:


event = 0                       # first event
features = events.dtype.names   # features that each event has


# In[5]:


print(f"Features for event: {event}")
for feat, value in zip(features, events[event]):
    print(f"{feat}: {value}")


# ## 1.2 PM signal data
# 
# - This section shows how to extract the PM signal data from the `Hits` data in an `HDF5` file.
# - To do so, you also need the `HitMapping` data
# - We index the `Hits` data as `Hits[HitsMapping[n]:HitsMapping[n+1]]` for event `n` to get the relevant data

# In[6]:


# pm information for each hit
pm_info = hits.dtype.names
pm_info


# In[7]:


# hitmap[n] to hitmap[n+1] is the indexing for hits, where n = event #
event_pm_info = hits[hit_map[event]:hit_map[event+1]]
event_pm_info


# In[8]:


# this is how many hits event 1 recorded, it will be variable for each event
print(f"# of hits for event {event+1}: {len(event_pm_info)}")


# In[9]:


# info for pm hit 1/22 for event 1
print(f"PM info for event: {event}")
for info, value in zip(pm_info, event_pm_info[0]):
    print(f"{info}: {value}")


# ## 1.3 Mapping PM signals to $(x, y)$ positions

# - The PM position map file is saved as `rich_pmt_positions.dat` in the `RICHPID` repo in the `tools` folder.
# - Bob has provided a tool to parse this file to `npy` format.  
# - To get this file one needs to run: `python conf_parser.py rich_pmt_positions.dat rich_pmt_positions.npy` in the `RICHPID/tools/` directory.
# - We will see that there are 1,952 PM signals with $(x, y)$ locations accross two mirrors.
# - We will use a function provided by Bob to help index the position map to extract the hit data that we need for each event.

# In[10]:


# this is our full map of (x, y) locations that the light hits
position_map = np.load("/home/nico/RICHPID/tools/rich_pmt_positions.npy")
position_map.shape


# In[11]:


# this is the first PM in the map
coord = position_map[1]

# pm location (x, y, mirror)
print("First pm in position map")
print(f"x:\t {coord[0]}")
print(f"y:\t {coord[1]}")
print(f"mirror:\t {coord[2]}")


# In[12]:


# We need this function provided by Bob to map the hits from our PM info for each event.

def compute_seq_id(disk_id, up_dw_id, sc_id, pm_id, or_id=0):
    """Compute the RICH PMT sequence ID"""
    if or_id < 1:
        seq_id = sc_id * 8 + pm_id + up_dw_id * 61 * 8 + disk_id * 61 * 8 * 2
    else:
        seq_id = 61 * 8 * 2 * 2 + sc_id + up_dw_id * 61 + disk_id * 61 * 2
    return seq_id


# In[13]:


# this is the pm info for the first hit for the first event
single_hit_pm_info = event_pm_info[0]

# this computes which value in the pm position map the hit occured at
pm_map_location = compute_seq_id(
    disk_id=single_hit_pm_info["disk_id"],
    up_dw_id=single_hit_pm_info["updowndisk_id"],
    sc_id=single_hit_pm_info["supercell_id"],
    pm_id=single_hit_pm_info["pmt_id"],
)

print(f"This hit occured at idx {pm_map_location} in the position map")


# In[14]:


# therefore the (x, y, mirror) information for this hit is
position_map[pm_map_location]


# ### 1.3.1 Getting all hit data for one event
# 
# - Here we simple use a `for` loop to get all the hit data for a single event.

# In[15]:


hits = []

for pm_info in event_pm_info:
    pm_idx = (
        compute_seq_id(
            disk_id=pm_info["disk_id"],
            up_dw_id=pm_info["updowndisk_id"],
            sc_id=pm_info["supercell_id"],
            pm_id=pm_info["pmt_id"]
        )
    )
    
    hits.append(position_map[pm_idx])
    
hits = np.array(np.array(hits))

# these are all the (x, y, mirror) locations for first first event
hits


# # Part 2: Some useful plots

# ## 2.1 Distribution of hit times
# - Theoretically the hit times should show one single peak

# In[16]:


hit_times = []

# get all the hit time data for the first event
for pm_info in event_pm_info:
    hit_times.append(pm_info["hit_time"])

plt.figure(figsize=(12, 8))
sns.kdeplot(hit_times)
plt.title(f"Distribution of hit times for event {event+1}", fontsize=20)
plt.xlabel("Time", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
("")


# - For this event, we can see below that the first two hits are very far away in time from the remaining hits, and are therefore noise.  
# - Our ML algorithm will need to deal with this noise.  We can also identify this based on `chod_time`, since we want our hit times to be close to the `chod_time`.

# In[17]:


sorted(hit_times)


# ## 2.2 CHOD time vs. Event Time
# - We want event times to be close to CHOD time.

# In[18]:


chod_time = events["chod_time"][0]

plt.figure(figsize=(12, 8))
plt.hlines(1,min(hit_times)-5,max(hit_times)+5, colors="black")  # Draw a horizontal line
plt.eventplot(hit_times, orientation='horizontal', colors='b')
plt.eventplot([chod_time], orientation='horizontal', colors='red', linelengths=2)
plt.text(4, 2.05, "CHOD time", color="red", fontsize=12)
plt.title("Event times vs. CHOD time", fontsize=20)
plt.xlabel("Time", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks([])
plt.show()


# ## 2.3 Ring plot
# 
# - Below we plot the pm array and the hits for the first event

# In[19]:


# plotting function from Bob that plots the PMT array
def draw_pmt_pos(ax,pmt_pos):
    """
        Add circle patches corresponding to the PMT position to the Axes object ax
    """
    for i in pmt_pos:
        if i[2] == 0: # 0: Jura / 1: Salève, PMT disks are identical, we can pick either one. [TODO: CHECK!]
            ax.add_patch(plt.Circle((i[0],i[1]),1.0, color='black'))
    return ax


# In[20]:


radius = events[event]['ring_radius']
centre = events[event]['ring_centre_pos']

print(radius)
print(centre)


# In[21]:


fig = plt.figure(figsize=(7, 7))
ax = fig.subplots()
ax.set_aspect('equal')
ax.set_xlim(-350,350) # mm
ax.set_ylim(-350,350) # mm
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_title(f'Event 1 Dataset A')

draw_pmt_pos(ax, position_map)
plt.scatter(hits[:, 0], hits[:, 1])
ax.add_artist(plt.Circle((0, 0), radius, fill=False,color="red"))
("")


# # Part 3: Looking at all the features

# <div class="alert alert-warning">
# 
# Key challenges:
#     
# 1. The numbers of hits for each event will be variable, so the number of $(x, y)$ postions will be variable in length for each event.
# 2. The same logic above applies for hit times.
# 3. Track momentum and ring centre are biased by the sampling process, we will need to deal with this in modelling.
#     
# </div>

# ## 3.1: List of all raw features

# | Name                  | Description                           | Notes                                                    | Located in                                                           |
# |-----------------------|---------------------------------------|----------------------------------------------------------|----------------------------------------------------------------------|
# | `run_id`              | The experiment run                    | Metadata                                                 | `HDF5["Events"]`                                                     |
# | `burst_id`            | The particle burst                    | Metadata                                                 | `HDF5["Events"]`                                                     |
# | `track_id`            | The particle track                    | Metadata                                                 | `HDF5["Events"]`                                                     |
# | `track_momentum`      | Particle momentum                     | Single value                                             | `HDF5["Events"]`                                                     |
# | `chod_time`           | Time recorded by CHOD detector        | Single value                                             | `HDF5["Events"]`                                                     |
# | `ring_radius`         | Radius of fitted circle               | Single value                                             | `HDF5["Events"]`                                                     |
# | `ring_centre_pos`     | Centre of fitted circle               | Array of two values: [x, y]                              | `HDF5["Events"]`                                                     |
# | `ring_liklihood`      | Liklihood of three particles          | Array of three values: [particle1, particle2, particle3] | `HDF5["Events"]`                                                     |
# | `disk_id`             | Used to locate hits with position map | Single value, input to `compute_seq_id`                  | `HDF5["Hits"]` indexed by `HDF5["HitMapping"]`                       |
# | `pmt_id`              | Used to locate hits                   | Single value, input to `compute_seq_id`                  | `HDF5["Hits"]` indexed by `HDF5["HitMapping"]`                       |
# | `supercell_id`        | Used to locate hits                   | Single value, input to `compute_seq_id`                  | `HDF5["Hits"]` indexed by `HDF5["HitMapping"]`                       |
# | `updowndisk_id`       | Used to locate hits                   | Single value, input to `compute_seq_id`                  | `HDF5["Hits"]` indexed by `HDF5["HitMapping"]`                       |
# | `hit_time`            | Time of each hit                      | Single value, but variable length for each event!        | `HDF5["Hits"]` indexed by `HDF5["HitMapping"]`                       |
# | `hits (x, y, mirror)` | Actual coordinates and mirror hit     | Variable length for each event!                          | Get from `rich_pmt_positions.npy` with indices from `compute_seq_id` |

# In[22]:


event = 0


# ## 3.2 Data from `HDF5["Events"]`
# 
# - This is just re-reading in the `Events` data for event 1 into a dictionary.

# In[23]:


event_features = {k:v for k, v in zip(f['Events'].dtype.names, f['Events'][event])}


# In[24]:


event_features


# ## 3.3 Hit data from `HDF5["Hits"]`, `HDF5["HitMapping"]`, and `rich_pmt_positions.npy`
# 
# - Here we re-read in the `Hits` data for event 1 into a dictionary.

# In[25]:


position_map = np.load("/home/nico/RICHPID/tools/rich_pmt_positions.npy")
position_map.shape


# In[26]:


hit_coords = []
hit_data = f['Hits'][f['HitMapping'][event]:f['HitMapping'][event+1]]
hit_times = hit_data["hit_time"]

# generate all hits coordinates for this event
for hit in hit_data:
    pm_idx = (
        compute_seq_id(
            disk_id=hit["disk_id"],
            up_dw_id=hit["updowndisk_id"],
            sc_id=hit["supercell_id"],
            pm_id=hit["pmt_id"]
        )
    )
    
    hit_coords.append(position_map[pm_idx])

hit_coords = np.array(hit_coords)

# combine coordinates with times
hit_coords_times = np.c_[hit_coords, hit_times]

hit_coords_times

