from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import torch

from Config import my_POSSMConfig
config = my_POSSMConfig()

import pandas as pd
lag_delta = pd.to_timedelta(config.time_lag, unit='ms')

from tqdm import tqdm
import json
import os
os.makedirs("processed_data", exist_ok=True)

# Load dataset
dataset = NWBDataset("./000128/sub-Jenkins/", "*train", split_heldout=False)

channel_ids = dataset.data['spikes'].keys()
num_channel = len(channel_ids) # 182 channels
channel_id_to_idx = {raw_id: i for i, raw_id in enumerate(channel_ids)}

# 1. Slice spikes into bins
def spike_slice_bins(input):
    time_length, num_channel = input.shape
    active_spike = [0]*(time_length//config.bin_size + 1)

    time_idxs, chan_idxs = np.nonzero(input)
    bin_indices = time_idxs // config.bin_size
    offsets = time_idxs % config.bin_size

    for spike in range(len(time_idxs)):
        bin_idx = bin_indices[spike]
        cid = chan_idxs[spike]
        offset = offsets[spike]
        if active_spike[bin_idx] == 0:
            active_spike[bin_idx] = [(cid, offset)]
        else:
            active_spike[bin_idx].append((cid, offset))

    return active_spike

# 2. Slice datasets according to trial_id
sliced_trials = []
max_bin = 0
max_token = 0
max_time_length = 0

for index, row in tqdm(dataset.trial_info.iterrows()):
    trial_id = row['trial_id']
    start_time = row['start_time']
    end_time = row['end_time']

    trial_spike_df = dataset.data.loc[start_time:end_time]["spikes"]
    trial_vel_df = dataset.data.loc[start_time+lag_delta:end_time+lag_delta]["hand_vel"]
    
    s_vals = trial_spike_df.values
    v_vals = trial_vel_df.values
    min_len = min(len(s_vals), len(v_vals))
    s_vals = s_vals[:min_len]
    v_vals = v_vals[:min_len]

    spikes_has_nan = np.isnan(s_vals).any(axis=1)
    vel_has_nan = np.isnan(v_vals).any(axis=1)
    valid_mask = ~spikes_has_nan & ~vel_has_nan

    spikes = s_vals[valid_mask] # shape: [time_length, channels]
    vel = v_vals[valid_mask] # shape: [time_length, 2] i.e. [[x, y], [x, y], ...]
    active_spike = spike_slice_bins(spikes) # shape: length is num_bins, each element is a list of (channel_id, offset), or 0 if no spike
    trial_max_token = max([len(bin) for bin in active_spike if bin != 0])

    sliced_trials.append({
        'trial_id': trial_id,
        'spikes': active_spike,  
        'vel': vel         
    })

    max_bin = max(max_bin, len(active_spike))
    max_token = max(max_token, trial_max_token)
    max_time_length = max(max_time_length, len(vel))

torch.save(sliced_trials, "processed_data/sliced_trials.pt")


meta_data = {
    "channel_ids": list(channel_ids),
    "num_channel": int(num_channel),
    "channel_id_to_idx": channel_id_to_idx,
    "num_bin_each_channel": {trial['trial_id']:len(trial["spikes"]) for trial in sliced_trials}, 
    "max_bin": max_bin, 
    "max_token": max_token,
    "max_time_length": max_time_length
}

with open("processed_data/meta_data.json", "w") as f:
    json.dump(meta_data, f, indent=4)