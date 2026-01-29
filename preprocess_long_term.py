import mat73
import numpy as np
import torch
import os
import json
import glob
from tqdm import tqdm
from Config import my_POSSMConfig

# 初始化配置
config = my_POSSMConfig()

# 路径设置
DATA_ROOT = 'long_term_data/Chewie_CO_2016'
OUTPUT_DIR = 'long_term_data/Chewie_processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def spike_counts_to_possm_format(spike_counts, bin_size_ms):
    """
    将 Dense 的 spike_counts 矩阵转换为 POSSM 需要的 Sparse 格式。
    
    Args:
        spike_counts: (T_time, N_channels) 矩阵，假设每一行代表 1ms (或其他细粒度单位)
        bin_size_ms: POSSM 的 bin 大小 (如 50ms)
    
    Returns:
        active_spike: List[List[Tuple] | 0]. 长度为 T_time // bin_size_ms。
                      每个元素是一个列表 [(channel_id, offset_in_bin), ...]
    """
    total_time, num_channels = spike_counts.shape
    num_bins = int(np.ceil(total_time / bin_size_ms))
    active_spike = [0] * num_bins
    
    # 获取所有非零 spike 的位置 (time_idx, channel_idx)    
    rows, cols = np.nonzero(spike_counts)
    bin_indices = rows // bin_size_ms
    offsets = rows % bin_size_ms
    for i in range(len(rows)):
        bin_idx = bin_indices[i]
        chan_idx = cols[i]
        offset = offsets[i]
        if active_spike[bin_idx] == 0:
            active_spike[bin_idx] = [(chan_idx, offset)]
        else:
            active_spike[bin_idx].append((chan_idx, offset))
    return active_spike

def process_single_session(file_path, session_idx):
    print(f"Processing Session {session_idx}: {os.path.basename(file_path)}")
    
    # 1. 加载数据
    data_dict = mat73.loadmat(file_path)
    xds = data_dict['xds']
    
    # 获取基础元数据
    # Chewie 数据集 bin_width 通常是 0.001 (1ms)
    raw_bin_width = xds['bin_width'] if 'bin_width' in xds else 0.001
    if np.abs(raw_bin_width-0.001) > 0.0001:
        print(f"Warning: Raw bin width is {raw_bin_width}s, expected 0.001s. Results may be scaled.")
    
    # 时间 lag 转换为 step (假设 1ms 分辨率)
    lag_steps = int(0) # e.g. 80ms -> 80 steps
    
    # 提取全量数据
    # 注意：xds['spike_counts'] 可能是 (Time, Channels)
    full_spikes = xds['spike_counts']
    num_channel = full_spikes.shape[1]
    full_vel = xds['curs_v']
    full_time = xds['time_frame']
    
    # 提取 Trial 信息
    trial_starts = xds['trial_start_time']
    trial_ends = xds['trial_end_time']
    trial_results = xds['trial_result'] # 字符串列表，如 ['R', 'A', 'F', ...] 或长字符串
    
    # 处理 trial_result 可能是单个长字符串或 list 的情况
    if isinstance(trial_results, str):
        trial_results = list(trial_results)
    
    sliced_trials = []
    
    # 遍历所有 Trial
    if len(trial_starts) != len(trial_ends):
        print(f"Warning: trial_starts and trial_ends have different lengths: {len(trial_starts)} != {len(trial_ends)}")
        return None

    max_bin = 0
    max_token = 0
    max_time_length = 0

    for i in tqdm(range(len(trial_starts)), desc="Slicing Trials"):
        # 1. 筛选成功 Trial ('R' = Reward)
        if trial_results[i] != 'R':
            continue
            
        t_start = trial_starts[i]
        t_end = trial_ends[i]
        
        # 2. 找到时间索引
        # 假设 time_frame 是连续且均匀的，可以用 searchsorted 或直接计算
        # 使用 searchsorted 更稳健
        idx_start = np.searchsorted(full_time, t_start)
        idx_end = np.searchsorted(full_time, t_end)
        
        # 3. 切片
        # 神经数据: t_start 到 t_end
        # 运动数据: t_start + lag 到 t_end + lag (模拟运动延迟)
        
        # 确保索引不越界
        if idx_end + lag_steps >= len(full_vel):
            continue
            
        trial_spikes = full_spikes[idx_start:idx_end, :]
        trial_vel = full_vel[idx_start+lag_steps : idx_end+lag_steps, :]
        
        # 长度对齐 (以防万一)
        min_len = min(len(trial_spikes), len(trial_vel))
        trial_spikes = trial_spikes[:min_len]
        trial_vel = trial_vel[:min_len]
        
        # 4. 转换为 POSSM 格式
        # Input: (T_ms, N_ch) -> Output: List of bins (50ms), each containing [(ch, off), ...]
        possm_spikes = spike_counts_to_possm_format(trial_spikes, config.bin_size)
        
        # 5. 保存
        # 注意: 这里的 vel 保持了原始分辨率 (1ms)，这是 POSSM Output Decoder 需要的
        sliced_trials.append({
            'trial_id': i,
            'spikes': possm_spikes, # [[(cid, offset), (cid, offset), ...], [(cid, offset), (cid, offset), ...], ...]
            'vel': trial_vel        # Numpy Array (T_ms, 2)
        })

        max_bin = max(max_bin, len(possm_spikes))
        max_token = max(max_token, max([len(bin) for bin in possm_spikes if bin != 0]))
        max_time_length = max(max_time_length, len(trial_vel))

    if len(sliced_trials) == 0:
        print(f"Warning: No valid trials found in {file_path}")
        return None

    # 计算该 Session 的统计量 (Mean, Std) 用于后续漂移分析中的归一化
    all_vels = np.concatenate([t['vel'] for t in sliced_trials], axis=0)
    vel_mean = np.mean(all_vels, axis=0)
    vel_std = np.std(all_vels, axis=0)

    print(f"Velocity Mean: {vel_mean}")
    print(f"Velocity Std: {vel_std}")
    
    # 保存该 Session 的数据
    save_folder = os.path.join(OUTPUT_DIR, f'session_{session_idx}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, 'sliced_trials.pt')
    torch.save(sliced_trials, save_path)
    
    meta_data = {
        "num_channel": num_channel,
        "max_bin": max_bin,
        "max_token": max_token,
        "max_time_length": max_time_length,
        "vel_mean": vel_mean.tolist(), # [mean_x, mean_y]
        "vel_std": vel_std.tolist()    # [std_x, std_y]
    }

    with open(os.path.join(save_folder, 'meta_data.json'), 'w') as f:
        json.dump(meta_data, f, indent=4)

def main():
    # 1. 查找所有 .mat 文件
    # 假设文件名中包含日期，排序以保证时间顺序 (Chewie_20160927_001.mat)
    mat_files = sorted(glob.glob(os.path.join(DATA_ROOT, '*.mat')))
    
    if not mat_files:
        print(f"No .mat files found in {DATA_ROOT}")
        return

    # 2. 逐个处理 Session
    for idx, f_path in enumerate(mat_files):
        process_single_session(f_path, idx)
        
if __name__ == "__main__":
    main()