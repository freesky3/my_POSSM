import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import json
meta_data = json.load(open("processed_data/meta_data.json", "r"))
max_bin = meta_data["max_bin"]
max_token = meta_data["max_token"]

def pad_spikes(trail_spikes, max_bin = max_bin, max_token = max_token): 
    trail_num = len(trail_spikes)
    padded_spikes = torch.zeros(trail_num, max_bin, max_token, 2, dtype=torch.int64)
    mask = torch.zeros(trail_num, max_bin, max_token, dtype=torch.bool)
    for i in range(trail_num):
        for j in range(max_bin):
            if j >= len(trail_spikes[i]) or not isinstance(trail_spikes[i][j], tuple):
                padded_spikes[i, j] = torch.tensor([[0, 0]])
                mask[i, j] = True
                continue
            for k in range(max_token):
                if k >= len(trail_spikes[i][j]):
                    padded_spikes[i, j, k] = torch.tensor([0, 0])
                    mask[i, j, k] = True
                else:
                    padded_spikes[i, j, k] = torch.tensor(trail_spikes[i][j][k])
                    mask[i, j, k] = False
                    
    return padded_spikes, mask


class my_dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, path):
        self.dataset = torch.load(path, weights_only=False)
        self.length_spike = torch.tensor([len(x["spikes"]) for x in self.dataset], dtype=torch.int64)
        self.lengths_vel = torch.tensor([len(x["vel"]) for x in self.dataset], dtype=torch.int64)
        # padded_spikes: (trial_num, max_bin, max_token, 2) 2 is [channel_id, offset]
        # mask_spikes: (trial_num, max_bin, max_token) True means padding
        self.padded_spikes, self.mask_spikes = pad_spikes([x["spikes"] for x in self.dataset])
        self.padded_vel = pad_sequence([torch.as_tensor(x["vel"]) for x in self.dataset], batch_first=True, padding_value=0.0) # (trial_num, max_time_length, 2)

    def __getitem__(self, idx):
        spike = self.padded_spikes[idx] # shape: [max_bin, max_token, 2]
        mask_spike = self.mask_spikes[idx] # shape: [max_bin, max_token]
        lengths_spike = self.length_spike[idx] # shape: [1]
        lengths_vel = self.lengths_vel[idx] # shape: [1]
        vel = self.padded_vel[idx] # shape: [max_time_length, 2]
        return spike, vel, mask_spike, lengths_spike, lengths_vel

    def __len__(self):
        return len(self.dataset)

def get_dataloader(data_dir="processed_data/sliced_trials.pt", batch_size=16, n_workers=0):
    """Generate dataloader"""
    dataset = my_dataset(data_dir)
    # Split dataset into training dataset and validation dataset
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
    )

    return train_loader, valid_loader