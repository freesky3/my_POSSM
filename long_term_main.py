from Config import my_POSSMConfig
config = my_POSSMConfig()

from tqdm import tqdm
from Dataloader import get_dataloader
from Model import my_POSSM
import torch
from torch.utils.tensorboard import SummaryWriter

import json
meta_data = json.load(open("long_term_data/Chewie_processed/session_0/meta_data.json", "r"))
VEL_MEAN = torch.tensor(meta_data["vel_mean"], dtype=torch.float32)
VEL_STD = torch.tensor(meta_data["vel_std"], dtype=torch.float32)

hyperparam = {
    "seed": 42,
    "num_epochs": 300,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "patience": 20,
    "log_dir": "./long_term_log",
    "model_path": "./long_term_model.pth",
}

import numpy as np
import random
import os

def set_seed(seed):
    """set the seed"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def masked_mse_loss(output, target, lengths):
    """
    Args:
        output: (batch_size, max_time_length, 2)
        target: (batch_size, max_time_length, 2)
        lengths: (batch_size) - 存储每个样本的有效长度
    """
    # 1. 生成基础掩码 (batch_size, max_time_length)
    batch_size, max_time, dim = output.shape
    device = output.device
    
    # torch.arange(max_time) 生成 [0, 1, 2, ..., max_time-1]
    # 利用广播机制与 lengths 比较
    mask = torch.arange(max_time, device=device).expand(batch_size, max_time) < lengths.unsqueeze(1)
    
    # 2. 将掩码扩展到特征维度 (batch_size, max_time_length, 2)
    # 增加最后一个维度并复制
    mask = mask.unsqueeze(-1).expand_as(output)
    
    # 3. 计算平方损失
    squared_diff = (output - target) ** 2
    
    # 4. 应用掩码并求平均
    # 只计算 mask 为 True 的部分的均值
    masked_squared_diff = squared_diff * mask.float()
    num_valid_elements = mask.sum()
    
    loss = masked_squared_diff.sum() / num_valid_elements
    return loss


def train_one_epoch(model, loader, optimizer, criterion, device, writer, epoch):
    model.train()
    mean_tensor = VEL_MEAN.to(device)
    std_tensor = VEL_STD.to(device)
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    for spike, bin_mask, spike_mask, vel, vel_lens in pbar:
        spike, bin_mask, spike_mask, vel, vel_lens = spike.to(device), bin_mask.to(device), spike_mask.to(device), vel.to(device), vel_lens.to(device)
        vel_lens = vel_lens - (config.k_history-1)*config.bin_size
        max_time_length = vel_lens.max()

        optimizer.zero_grad()
        outputs = model(spike, bin_mask, spike_mask)
        outputs = outputs[:, :max_time_length, :] # (batch_size, max_time_length-(config.k_history-1)*config.bin_size, 2)
        normalized_vel = (vel - mean_tensor) / std_tensor
        tru_norm_vel = normalized_vel[:, (config.k_history-1)*config.bin_size:, :]
        loss = criterion(outputs, tru_norm_vel, vel_lens)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(loader)
    
    # 写入 TensorBoard
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    
    return epoch_loss

@torch.no_grad()
def validate(model, loader, criterion, device, writer, epoch):
    model.eval()
    mean_tensor = VEL_MEAN.to(device)
    std_tensor = VEL_STD.to(device)
    running_loss = 0.0
    
    # 注意：这里 loader 返回的数据解包要和 dataset 对应，
    # 你的 dataset 似乎返回 5 个值，这里需要全部接收
    for spike, bin_mask, spike_mask, vel, vel_lens in tqdm(loader, desc=f"Validating", leave=True):
        spike, bin_mask, spike_mask, vel, vel_lens = spike.to(device), bin_mask.to(device), spike_mask.to(device), vel.to(device), vel_lens.to(device)
        vel_lens = vel_lens - (config.k_history-1)*config.bin_size
        max_time_length = vel_lens.max()
        
        outputs = model(spike, bin_mask, spike_mask)
        outputs = outputs[:, :max_time_length, :] # (batch_size, max_time_length-(config.k_history-1)*config.bin_size, 2)
        norm_vel = (vel - mean_tensor) / std_tensor
        tru_norm_vel = norm_vel[:, (config.k_history-1)*config.bin_size:, :]
        loss = criterion(outputs, tru_norm_vel, vel_lens) # 使用同样的 masked_mse_loss
        
        running_loss += loss.item()
        
    val_loss = running_loss / len(loader)
    
    writer.add_scalar('Loss/Valid', val_loss, epoch)
    
    return val_loss



def main():
    writer = SummaryWriter(log_dir=hyperparam['log_dir'])

    best_val_loss = float('inf')
    early_stopping_counter = 0

    set_seed(hyperparam['seed'])
    train_loader, valid_loader = get_dataloader(data_dir="long_term_data/Chewie_processed/session_0/sliced_trials.pt")
    num_channel = meta_data["num_channel"]
    model = my_POSSM(config, num_channel=num_channel).to(hyperparam['device'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparam['learning_rate'], weight_decay=hyperparam['weight_decay'])
    criterion = masked_mse_loss

    for epoch in range(hyperparam['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, hyperparam['device'], writer, epoch)
        val_loss = validate(model, valid_loader, criterion, hyperparam['device'], writer, epoch)

        print(f'Epoch {epoch+1}/{hyperparam["num_epochs"]}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 1. 检查是否是最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # 重置计数器
            
            # 保存最佳模型权重
            torch.save(model.state_dict(), hyperparam['model_path'])
            print(f"--> Best model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")
            
        else:
            early_stopping_counter += 1
            print(f"--> No improvement. Patience: {early_stopping_counter}/{hyperparam['patience']}")

        # 2. 检查是否触发早停
        if early_stopping_counter >= hyperparam['patience']:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break



if __name__ == "__main__":
    main()
