import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from Config import my_POSSMConfig
config = my_POSSMConfig()
k_history = config.k_history

class WindowedDataset(Dataset):
    def __init__(self, path, k_history=k_history):
        raw_data = torch.load(path, weights_only=False)
        self.k_history = k_history
        self.samples = [] # 存储索引 (trial_idx, start_bin_idx)
        self.data_ref = [] # 保存原始数据引用，避免重复拷贝内存

        # 预处理：构建滑窗索引
        for trial_idx, trial in enumerate(raw_data):
            spikes = trial["spikes"] # # shape: length is num_bins, each element is a list of (channel_id, offset), or 0 if no spike
            vel = torch.as_tensor(trial["vel"], dtype=torch.float32) # [time_length, 2]

            num_bins = len(spikes)

            if num_bins >= k_history:
                self.data_ref.append({
                    "spikes": spikes, 
                    "vel": vel
                })
                # 记录每个窗口的起始位置
                # 这里的逻辑是：sliding window, stride=1
                # 如果希望不重叠，可以将 range 的步长设为 k_history
                current_ref_idx = len(self.data_ref) - 1
                for start_idx in range(num_bins - k_history + 1):
                    self.samples.append((current_ref_idx, start_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # samples: (trial_idx, start_bin_idx)
        ref_idx, start_idx = self.samples[idx]
        trial_data = self.data_ref[ref_idx]
        
        end_idx = start_idx + self.k_history
        
        # 1. 获取 Velocity 片段 (Time, 2)
        vel_window = torch.as_tensor(trial_data["vel"][(end_idx-1)*config.bin_size:(end_idx)*config.bin_size], dtype=torch.float32)
        
        # 2. 获取 Spikes 片段
        raw_spike_window = trial_data["spikes"][start_idx:end_idx] # shape: [k_history, **var_length**, 2]
        
        spike_seq_tensors = []
        for bin_spikes in raw_spike_window:
            valid_spikes = []
            if isinstance(bin_spikes, list):
                for s in bin_spikes:
                    valid_spikes.append(s)
            
            if len(valid_spikes) > 0:
                # 转换为 Tensor: (num_spikes, 2)
                spike_seq_tensors.append(torch.tensor(valid_spikes, dtype=torch.long))
            else:
                # 如果这一帧没有 spike，放一个空的 tensor (0, 2)
                spike_seq_tensors.append(torch.zeros((0, 2), dtype=torch.long))

        # 返回：
        # spike_seq_tensors: 一个长度为 k_history 的 List，里面每个元素是 Tensor(num_spikes, 2)
        # vel_window: k_history个bin中最后一个bin的Velocity Tensor(time_length, 2)
        return spike_seq_tensors, vel_window

def ssm_collate_fn(batch):
    """
    自定义 Collate 函数
    Batch 输入结构: List of (spike_seq_tensors, vel_window)
    """
    batch_size = len(batch)
    
    # 获取时间窗口长度 (假设所有样本的 k_history 是固定的)
    k_history = len(batch[0][0]) 
    
    # --- 准备容器 ---
    flattened_spikes = []
    spike_lengths = []    # 记录每个 bin 的 spike 数量
    
    vel_list = []         # 收集所有 vel 用于 padding
    vel_lengths = []      # <--- 新增：记录每个样本 vel 的时间长度
    
    # --- 单次循环遍历 ---
    for item in batch:
        # 1. 解包
        spike_seq = item[0] # List[Tensor]
        vel = item[1]       # Tensor
        
        # 2. 收集 Spikes 信息
        flattened_spikes.extend(spike_seq)
        spike_lengths.extend([len(s) for s in spike_seq]) 

        # 3. 收集 Velocity 信息
        vel_list.append(vel)
        vel_lengths.append(len(vel)) # <--- 记录当前样本的时间步长

    # --- 处理 Spikes ---
    # shape: (Batch * k_history, Max_Spikes, 2)
    padded_flat = pad_sequence(flattened_spikes, batch_first=True, padding_value=0)
    
    # Reshape 回 (Batch, k_history, Max_Spikes, 2)
    max_spikes_in_batch = padded_flat.size(1)
    final_spikes = padded_flat.view(batch_size, k_history, max_spikes_in_batch, 2)
    
    # Spike Lengths: (Batch, k_history)
    spike_lengths_tensor = torch.tensor(spike_lengths, dtype=torch.long).view(batch_size, k_history)
    
    # --- 处理 Velocity ---
    # 使用 pad_sequence 确保即使 vel 长度有微小差异也能对齐
    # batch_first=True -> (Batch, Time, 2)
    batch_vel = pad_sequence(vel_list, batch_first=True, padding_value=0)
    vel_lengths_tensor = torch.tensor(vel_lengths, dtype=torch.long)
    
    # final_spikes: (batch_size, k_history, max_spikes_in_batch, 2)
    # spike_lengths_tensor: (batch_size, k_history)
    # batch_vel: (batch_size, time_length, 2)
    # vel_lengths_tensor: (batch_size)
    return final_spikes, batch_vel, spike_lengths_tensor, vel_lengths_tensor

def get_dataloader(data_dir="processed_data/sliced_trials.pt", batch_size=16, n_workers=0):
    """Generate dataloader with custom collate"""
    dataset = WindowedDataset(data_dir, k_history=k_history)
    
    # Split dataset
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    generator=torch.Generator().manual_seed(config.seed) # 建议固定随机种子以便复现
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=ssm_collate_fn # <--- 关键：挂载自定义 collate
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=ssm_collate_fn # <--- 关键：挂载自定义 collate
    )

    return train_loader, valid_loader