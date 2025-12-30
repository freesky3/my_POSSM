import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class POSSM_Backbone_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=1, dropout=0.0):
        """
        初始化 GRU Backbone.
        
        - input_dim: 对应 Input Cross-Attention 输出的 config.num_latents * config.embed_dim
        - hidden_dim: GRU 的隐藏层维度
        - num_layers: GRU 层数
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 使用 PyTorch 标准 GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, z_t, bin_mask, h_prev=None):
        """
        前向传播
        
        Args:
            z_t: 来自 Input Cross-Attention 的潜在向量。
                 Shape: (batch_size, max_bin, num_latents, embed_dim)
            bin_mask: 布尔掩码，标识每个序列的有效位置。
                Shape: (batch_size, max_bin)，True表示有效位置，False表示padding
            h_prev: 上一时刻的隐藏状态。
                    Shape: (num_layers, Batch, hidden_dim)
                    如果是序列开始，可以为 None (默认为全0)。
        
        Returns:
            output: GRU 所有时间步的输出 (Batch, max_bin, hidden_dim)
        """
        # output 包含序列中每个时间步的 hidden state (用于 Output Attention), shape: (batch_size, max_bin, hidden_dim)
        batch_size, max_bin = z_t.shape[0], z_t.shape[1]
        z_t = z_t.view(batch_size, max_bin, -1)
        
        # 从 bin_mask 计算每个序列的实际长度
        # bin_mask: (batch_size, max_bin)，对每个序列求和得到有效长度
        lengths = bin_mask.sum(dim=1)  # Shape: (batch_size,)
        lengths_cpu = lengths.cpu().to(torch.int64)
        
        # 使用 pack_padded_sequence 处理变长序列，提高计算效率
        packed_input = pack_padded_sequence(z_t, lengths_cpu, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed_input, h_prev)
        
        # 将打包的输出重新填充为固定长度，便于后续处理
        output_padded, _ = pad_packed_sequence(output, batch_first=True, total_length=max_bin)
        
        return output_padded