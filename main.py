from Config import my_POSSMConfig
config = my_POSSMConfig()

from tqdm import tqdm
from Dataloader import get_dataloader
from Model import my_POSSM
import torch
from torch.utils.tensorboard import SummaryWriter

hyperparam = {
    "batch_size": 256,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_dir": "./log",
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
        output: (batch_size, bin_size, feature_dim) -> 模型输出
        target: (batch_size, time_length, feature_dim) -> 真实标签
        lengths: (batch_size,) -> 每个样本的真实时间步长
    """
    # 1. 维度对齐 (Alignment)
    # 取两者中较短的时间长度，防止 shape mismatch
    # 通常情况下，如果是 Seq2Seq，output 和 target 长度应该一致，但在你的 collate_fn 中可能会有微小差异
    min_len = min(output.size(1), target.size(1))
    
    # 截取有效部分
    output = output[:, :min_len, :]
    target = target[:, :min_len, :]
    
    # 2. 生成 Mask (Boolean Mask)
    # shape: (batch_size, min_len)
    batch_size = output.size(0)
    device = output.device
    
    # 创建位置索引 [0, 1, 2, ... min_len-1]
    range_tensor = torch.arange(min_len, device=device).unsqueeze(0) # (1, min_len)
    # 扩展 lengths 以便比较
    lengths_expanded = lengths.unsqueeze(1).to(device) # (batch_size, 1)
    
    # 生成 Mask: 如果位置索引 < 真实长度，则为 True (有效)，否则为 False (Padding)
    mask = range_tensor < lengths_expanded # (batch_size, min_len)
    
    # 3. 计算 MSE (Squared Error)
    # shape: (batch_size, min_len, feature_dim)
    loss = (output - target) ** 2
    
    # 4. 应用 Mask
    # 将 mask 扩展到 feature 维度: (batch_size, min_len, 1)
    mask_expanded = mask.unsqueeze(-1).float()
    
    # 只保留有效部分的 loss，padding 部分变为 0
    masked_loss = loss * mask_expanded
    
    # 5. 计算平均值 (Reduction)
    # 分子：所有有效位置的 loss 之和
    sum_loss = masked_loss.sum()
    
    # 分母：有效元素的总个数 (Time * Feature_dim)
    # 注意：这里要乘以 output.size(-1) (即 feature_dim=2)，因为 loss 是对每个特征都算了
    num_active_elements = mask_expanded.sum() * output.size(-1)
    
    # 防止分母为 0 (虽然极不可能)
    mse_loss = sum_loss / (num_active_elements + 1e-8)
    
    return mse_loss

def train_one_epoch(model, loader, optimizer, criterion, device, writer, epoch):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    for spike, vel, spike_lengths, vel_lengths in pbar:
        spike, vel = spike.to(device), vel.to(device)
        spike_lengths, vel_lengths = spike_lengths.to(device), vel_lengths.to(device)

        optimizer.zero_grad()
        outputs = model(spike, spike_lengths)
        
        loss = criterion(outputs, vel, vel_lengths)
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
    running_loss = 0.0
    
    # 注意：这里 loader 返回的数据解包要和 dataset 对应，
    # 你的 dataset 似乎返回 5 个值，这里需要全部接收
    for spike, vel, spike_lengths, vel_lengths in loader:
        spike, vel = spike.to(device), vel.to(device)
        spike_lengths, vel_lengths = spike_lengths.to(device), vel_lengths.to(device)
        
        outputs = model(spike, spike_lengths)
        loss = criterion(outputs, vel, vel_lengths) # 使用同样的 masked_mse_loss
        
        running_loss += loss.item()
        
    val_loss = running_loss / len(loader)
    
    writer.add_scalar('Loss/Valid', val_loss, epoch)
    
    return val_loss



def main():
    writer = SummaryWriter(log_dir=hyperparam['log_dir'])

    set_seed(config.seed)
    train_loader, valid_loader = get_dataloader()
    model = my_POSSM(config).to(hyperparam['device'])

    optimizer = torch.optim.SGD(model.parameters(), lr=hyperparam['learning_rate'], momentum=0.9) 
    criterion = masked_mse_loss

    for epoch in range(hyperparam['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, hyperparam['device'], writer, epoch)
        val_loss = validate(model, valid_loader, criterion, hyperparam['device'], writer, epoch)

        print(f'Epoch {epoch+1}/{hyperparam["num_epochs"]}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')



if __name__ == "__main__":
    main()
