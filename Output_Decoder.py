import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from RoPE import RotaryEmbedding

from Config import my_POSSMConfig
config = my_POSSMConfig()

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    batch_size, max_bin, max_token, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, :, None, :]
        .expand(batch_size, max_bin, max_token, num_key_value_heads, n_rep, head_dim)
        .reshape(batch_size, max_bin, max_token, num_key_value_heads * n_rep, head_dim)
    )

class POSSMOutputDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.gru_hidden_size  # Backbone 输出维度
        self.embed_dim = config.embed_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.n_rep = self.num_attention_heads // self.num_key_value_heads
        
        # 论文参数
        self.k_history = config.k_history # 论文提及 k=3，向后看3个chunk
        self.bin_size = config.bin_size # 50ms
        
        # Projections
        # Q: 我们生成 bin_size 个 query，去查询 k_history 个 hidden states
        self.q_proj = nn.Linear(self.embed_dim, self.num_attention_heads * self.head_dim, bias=False) 
        self.k_proj = nn.Linear(self.input_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.input_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.embed_dim, bias=False)
        
        # Learnable Queries: 对应一个 bin 内的每一个细粒度时间点 (例如 0ms, 1ms, ..., 49ms)
        # Shape: (bin_size, embed_dim)
        self.output_queries = nn.Parameter(torch.randn(self.bin_size, self.embed_dim))
        
        # Readout Head: 映射到行为变量 (例如 2D 速度)
        self.readout = nn.Linear(self.embed_dim, 2) 

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, h, freqs_cos, freqs_sin):
        """
        Args:
            h: GRU Hidden States. Shape: (Batch, max_bin, input_size)
            freqs_cos, freqs_sin: RoPE 预计算频率 (用于 Query 的时间编码)
                shape: (max_bin, head_dim)
        
        Returns:
            predictions: (Batch, max_bin * bin_size, 2) -> 展平的高频预测
        """
        batch_size, max_bin, _ = h.shape
        
        # ==========================================
        # 1. 准备 Keys 和 Values (滑动窗口)
        # ==========================================
        # 我们需要为每个时间步 t 构造上下文 [h_{t-k+1}, ..., h_t]
        # 先在时间维度 padding，以便 t=0 时也能取到前 k 个 (padding 为 0)
        padding = torch.zeros(batch_size, self.k_history - 1, self.input_size, device=h.device)
        h_padded = torch.cat([padding, h], dim=1) # (Batch, max_bin + k - 1, Dim)
        
        # 使用 unfold 构建滑动窗口
        # unfold(dimension, size, step)
        # result shape: (Batch, max_bin, Dim, k) -> 调整为 (Batch, max_bin, k, Dim)
        h_windowed = h_padded.unfold(1, self.k_history, 1).permute(0, 1, 3, 2)
        
        # 将 Batch 和 max_bin 合并，视为独立的 Attention 样本
        # KV Shape: (Batch * max_bin, k, input_size)
        kv_input = h_windowed.reshape(batch_size * max_bin, self.k_history, self.input_size)
        
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)
        
        # Reshape for Multi-head: (Batch, max_bin, k, Heads, Head_Dim)
        k = k.view(batch_size, max_bin, self.k_history, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, max_bin, self.k_history, self.num_key_value_heads, self.head_dim)

        # ==========================================
        # 2. 准备 Queries (Learnable + RoPE)
        # ==========================================
        # Q Shape: (Batch, max_bin, bin_size, embed_dim)
        q_base = self.output_queries.unsqueeze(0).expand(batch_size, max_bin, -1, -1)
        q = self.q_proj(q_base)
        
        # Reshape: (Batch, max_bin, bin_size, Heads, Head_Dim)
        q = q.view(batch_size, max_bin, self.bin_size, self.num_attention_heads, self.head_dim)
        
        # --- Apply RoPE to Queries ---
        # 论文指出 Query 编码了 timestamp。
        # 这里我们需要为每个 bin 内的 P 个点生成对应的 offset。
        # 简单起见，所有 bin 内部的时间结构是一样的 (0~49)，所以 offset 是固定的。
        # 注意：你需要确保传入的 freqs_cos/sin 长度足够覆盖 points_per_bin
        
        # 构造简单的 offset: 0, 1, ..., P-1
        offsets = torch.arange(self.bin_size, device=h.device)
        offsets = offsets.unsqueeze(0).expand(batch_size, max_bin, self.bin_size) # (Batch, max_bin, bin_size)
        
        q = RotaryEmbedding.apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin, offsets, is_decoder=True)
        q = q.transpose(2, 3) # (Batch, max_bin, Heads, bin_size, Head_Dim)
        k = repeat_kv(k, self.n_rep).transpose(2, 3) # (Batch, max_bin, Heads, k_History, Head_Dim)
        v = repeat_kv(v, self.n_rep).transpose(2, 3) # (Batch, max_bin, Heads, k_History, Head_Dim)
        # ==========================================
        # 3. Cross Attention
        # ==========================================
        # Scores: (batch_size, max_bin, Heads, bin_size, k_History)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Output: (batch_size, max_bin, Heads, bin_size, Head_Dim)
        attn_out = attn_weights @ v 
        
        # Merge Heads
        attn_out = attn_out.transpose(2, 3).reshape(batch_size, max_bin, self.bin_size, -1)
        
        # Output Projection
        z_out = self.o_proj(attn_out) # -> embed_dim
        z_out = self.dropout(z_out)

        # ==========================================
        # 4. Readout (Linear)
        # ==========================================
        # 映射到速度 (batch_size, max_bin, bin_size, 2)
        vel_pred = self.readout(z_out)
        
        # 展平回时间序列: (batch_size, max_bin * bin_size, 2)
        vel_pred = vel_pred.view(batch_size, -1, 2)
        
        return vel_pred