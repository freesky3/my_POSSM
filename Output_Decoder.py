import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from RoPE import RotaryEmbedding

from Config import my_POSSMConfig
config = my_POSSMConfig()

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    batch_size, k_history, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, k_history, num_key_value_heads, n_rep, head_dim)
        .reshape(batch_size, k_history, num_key_value_heads * n_rep, head_dim)
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
            h: GRU Hidden States. Shape: (Batch, k_history, gru_hidden_dim)
            freqs_cos, freqs_sin: RoPE 预计算频率 (用于 Query 的时间编码)
                shape: (bin_size, head_dim)
        
        Returns:
            predictions: (Batch, max_bin * bin_size, 2) -> 展平的高频预测
        """
        batch_size, k_history, _ = h.shape
        
        
        k = self.k_proj(h)
        v = self.v_proj(h)
        
        # Reshape for Multi-head: (Batch, k_history, Heads, Head_Dim)
        k = k.view(batch_size, k_history, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, k_history, self.num_key_value_heads, self.head_dim)

        # ==========================================
        # 2. 准备 Queries (Learnable + RoPE)
        # ==========================================
        # Q Shape: (Batch, bin_size, embed_dim)
        q_base = self.output_queries.unsqueeze(0).expand(batch_size, -1, -1)
        q = self.q_proj(q_base)
        
        # Reshape: (Batch, bin_size, Heads, Head_Dim)
        q = q.view(batch_size, self.bin_size, self.num_attention_heads, self.head_dim)
        
        # --- Apply RoPE to Queries ---
        # 论文指出 Query 编码了 timestamp。
        # 注意：你需要确保传入的 freqs_cos/sin 长度足够覆盖 points_per_bin

        q = RotaryEmbedding.apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin, is_decoder=True)
        q = q.transpose(1, 2) # (Batch, Heads, bin_size, Head_Dim)
        k = repeat_kv(k, self.n_rep).transpose(1, 2) # (Batch, Heads, k_history, Head_Dim)
        v = repeat_kv(v, self.n_rep).transpose(1, 2) # (Batch, Heads, k_history, Head_Dim)
        # ==========================================
        # 3. Cross Attention
        # ==========================================
        # Scores: (batch_size, Heads, bin_size, k_history)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Output: (batch_size, Heads, bin_size, Head_Dim)
        attn_out = attn_weights @ v 
        
        # Merge Heads
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, self.bin_size, -1)
        
        # Output Projection
        z_out = self.o_proj(attn_out) # (Batch, bin_size, embed_dim)
        z_out = self.dropout(z_out)

        # ==========================================
        # 4. Readout (Linear)
        # ==========================================
        # 映射到速度 (batch_size, bin_size, 2)
        vel_pred = self.readout(z_out)
        
        return vel_pred