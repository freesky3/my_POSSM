from itertools import chain, islice

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class my_dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, path):
        self.dataset = torch.load(path, weights_only=False)

    def __getitem__(self, idx):
        spike = self.dataset[idx]["spikes"]
        vel = self.dataset[idx]["vel"]
        return spike, vel

    def __len__(self):
        return len(self.dataset)




def pad_collate_fn(batch):
    trials, vel = zip(*batch)
    # pad spikes
    bins_per_trial = [len(trial) for trial in trials]
    unfold_spikes = list(chain.from_iterable(trials))
    new_unfold_spikes = [x if x != 0 else torch.empty(0, 2) for x in unfold_spikes]

    bin_seq_lens = torch.tensor([len(x) for x in new_unfold_spikes]) 
    it_len = iter(bin_seq_lens)
    bin_seq_lens_restored = [torch.tensor(list(islice(it_len, l))) for l in bins_per_trial]
    padded_seq_lens = pad_sequence(bin_seq_lens_restored, batch_first=True, padding_value=0)

    tensor_unfold_spikes = [x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in new_unfold_spikes]
    padded_unfold_spikes = pad_sequence(tensor_unfold_spikes, batch_first=True, padding_value=0)
    it = iter(padded_unfold_spikes)
    padded_spikes = [list(islice(it, trial_len)) for trial_len in bins_per_trial]

    torch_padded_spikes = [torch.stack(x) for x in padded_spikes]
    padded_bin = pad_sequence(torch_padded_spikes, batch_first=True, padding_value=0)
    
    trial_counts = torch.tensor(bins_per_trial)
    max_bins = padded_bin.size(1)
    bin_mask = torch.arange(max_bins)[None, :] < trial_counts[:, None]

    max_seq_len = padded_bin.size(2)
    seq_range = torch.arange(max_seq_len).view(1, 1, -1)
    len_target = padded_seq_lens.unsqueeze(-1)
    spike_mask = seq_range < len_target

    vel = [torch.as_tensor(v) for v in vel]
    vel = pad_sequence(vel, batch_first=True, padding_value=0.0) # (batch_size, max_time_length, 2)
    vel_lens = torch.tensor([len(x) for x in vel])
    # True: valid, False: padding value
    return padded_bin, bin_mask, spike_mask, vel, vel_lens


def get_dataloader(data_dir="processed_data/sliced_trials.pt", batch_size=8, n_workers=0):
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
        collate_fn=pad_collate_fn,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )

    return train_loader, valid_loader