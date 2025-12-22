from Config import my_POSSMConfig
config = my_POSSMConfig()

from tqdm import tqdm
from Dataloder import get_dataloader
from Model import my_POSSM
import torch
from torch.utils.tensorboard import SummaryWriter

hyperparam = {
    "seed": 42,
    "batch_size": 32,
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
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    for spike, vel, mask_spike, lengths_spike, lengths_vel in pbar:
        spike, vel = spike.to(device), vel.to(device)
        mask_spike, lengths_spike = mask_spike.to(device), lengths_spike.to(device)
        lengths_vel = lengths_vel.to(device)

        optimizer.zero_grad()
        outputs = model(spike, mask_spike, lengths_spike)
        
        loss = criterion(outputs, vel, lengths_vel)
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
    for spike, vel, mask_spike, lengths_spike, lengths_vel in loader:
        spike, vel = spike.to(device), vel.to(device)
        mask_spike, lengths_spike = mask_spike.to(device), lengths_spike.to(device)
        lengths_vel = lengths_vel.to(device)
        
        outputs = model(spike, mask_spike, lengths_spike)
        loss = criterion(outputs, vel, lengths_vel) # 使用同样的 masked_mse_loss
        
        running_loss += loss.item()
        
    val_loss = running_loss / len(loader)
    
    writer.add_scalar('Loss/Valid', val_loss, epoch)
    
    return val_loss



def main():
    writer = SummaryWriter(log_dir=hyperparam['log_dir'])

    set_seed(hyperparam['seed'])
    train_loader, valid_loader = get_dataloader()
    model = my_POSSM(config)

    optimizer = torch.optim.SGD(model.parameters(), lr=hyperparam['learning_rate'], momentum=0.9) 
    criterion = masked_mse_loss

    for epoch in range(hyperparam['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, hyperparam['device'], writer, epoch)
        val_loss = validate(model, valid_loader, criterion, hyperparam['device'], writer, epoch)

        print(f'Epoch {epoch+1}/{hyperparam["num_epochs"]}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')



if __name__ == "__main__":
    main()
