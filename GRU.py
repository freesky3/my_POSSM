import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

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

    def forward(self, z_t, h_prev=None):
        """
        前向传播
        
        Args:
            z_t: 来自 Input Cross-Attention 的潜在向量。
                 Shape: (batch_size, k_history, num_latents, embed_dim)
            lengths: 每个序列的长度
                Shape: (batch_size)
            h_prev: 上一时刻的隐藏状态。
                    Shape: (num_layers, Batch, hidden_dim)
                    如果是序列开始，可以为 None (默认为全0)。
        
        Returns:
            output: GRU 所有时间步的输出 (Batch, Seq_Len, hidden_dim)
            h_new: 更新后的隐藏状态 (num_layers, Batch, hidden_dim)
        """
        # output 包含序列中每个时间步的 hidden state (用于 Output Attention), shape: (batch_size, k_history, hidden_dim)
        # h_new 是序列最后一个时间步的 hidden state (用于传给下一个 chunk), shape: (num_layers, batch_size, hidden_dim)
        batch_size, k_history = z_t.shape[0], z_t.shape[1]
        z_t = z_t.view(batch_size, k_history, -1)
        output, h_new = self.gru(z_t, h_prev)
        
        return output, h_new