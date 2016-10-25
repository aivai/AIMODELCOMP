# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:41:50 2016

@author: aivai
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
from multiprocessing import Pool
import lims_utils
from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor



from allensdk.ephys.feature_extractor import EphysFeatureExtractor



from allensdk.core.cell_types_cache import CellTypesCache #Following jupyter notebook here
import pprint
pp = pprint.PrettyPrinter(indent=2)

ctc = CellTypesCache(manifest_file='cell_types/cell_types_manifest.json')

from allensdk.api.queries.cell_types_api import CellTypesApi
#%%

ephys_features = ctc.get_ephys_features()

data_set = ctc.get_ephys_data(464212183) # For one particular specimen (later find one that has all models, follows trend in plots)

#ct = CellTypesApi()
#cells = ct.list_cells(require_reconstruction=False)
#ct.save_ephys_data(cells[0]['476218657'], 'example.nwb')

exp_sweeps = []
exp_spike_times=[]

for sweep in data_set.get_sweep_numbers():
    if data_set.get_sweep_metadata(sweep)['aibs_stimulus_name'] == 'Noise 1':
        exp_sweeps.append(sweep)
        exp_spike_times.append(data_set.get_spike_times(sweep))

fig, axes = plt.subplots(2, 1, sharex=True)
     
for exp_sweep in exp_sweeps:
    sweep_data = data_set.get_sweep(exp_sweep)
    index_range = sweep_data["index_range"]
    i = sweep_data["stimulus"][0:index_range[1]+1] # in A
    v = sweep_data["response"][0:index_range[1]+1] # in V
    i *= 1e12 # to pA
    v *= 1e3 # to mV
    sampling_rate = sweep_data["sampling_rate"] # in Hz
    t = np.arange(0, len(v)) * (1.0 / sampling_rate)
    fx = EphysFeatureExtractor()
    stim_start = 1.0
    stim_duration = 1.0
    fx.process_instance("", v, i, t, stim_start, stim_duration, "")
    feature_data = fx.feature_list[0].mean
    plt.style.use('ggplot')

    axes[0].plot(t, v)
    axes[1].plot(t, i)
    axes[0].set_ylabel("mV")
    axes[1].set_ylabel("pA")
    axes[1].set_xlabel("seconds")
    
plt.show()

fig, axes = plt.subplots(1, 1)

for exp_sweep in exp_sweeps:
    sweep_data = exp_spike_times[exp_sweeps.index(exp_sweep)]
    n, bins, patches = plt.hist(sweep_data, 50, normed=1, alpha=0.75)
    
plt.show()

fig, axes = plt.subplots(1, 1)

for exp_sweep in exp_sweeps:
    sweep_data = exp_spike_times[exp_sweeps.index(exp_sweep)]
    index_range = sweep_data["index_range"]
    i = sweep_data["stimulus"][0:index_range[1]+1] # in A
    v = sweep_data["response"][0:index_range[1]+1] # in V
    i *= 1e12 # to pA
    v *= 1e3 # to mV
    sampling_rate = sweep_data["sampling_rate"] # in Hz
    t = np.arange(0, len(v)) * (1.0 / sampling_rate)
    fx = EphysFeatureExtractor()
    stim_start = 1.0
    stim_duration = 1.0
    fx.process_instance("", v, i, t, stim_start, stim_duration, "")
    feature_data = fx.feature_list[0].mean
    plt.style.use('ggplot')

    axes[0].scatter(t, v)
    axes[1].plot(t, i)
    axes[0].set_ylabel("mV")
    axes[1].set_ylabel("pA")
    axes[1].set_xlabel("seconds")
    
plt.show()
# Make plot of mean PSTH based on above data
# Spike time vectors are different lengths, how best to avg together?

# Next, plot predicted spike times for each model 



EphysFeatureExtractor.isicv()










#%%




#%%

NEURONAL_MODEL_TEMPLATES = {
    "glif_1": 395310469,
    "glif_2": 395310479,
    "glif_3": 395310475,
    "glif_4": 471355161,
    "glif_5": 395310498,
    "bp": 329230710,
    "baa": 491455321,
}

