import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from RoPE import RotaryEmbedding

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    batch_size, k_history, max_spikes_in_batch, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, :, None, :]
        .expand(batch_size, k_history, max_spikes_in_batch, num_key_value_heads, n_rep, head_dim)
        .reshape(batch_size, k_history, max_spikes_in_batch, num_key_value_heads * n_rep, head_dim)
    )

class POSSMCrossAttention(nn.Module):
    """
    POSSM Input Cross-Attention Module.
    
    This module implements the POYO-style encoder described in the POSSM paper.
    It compresses a variable-length sequence of spike tokens (Keys/Values) into 
    a fixed-size latent representation using learnable Queries.
    """
    def __init__(self, config):
        super().__init__()
        # Use existing config, default to 1 latent if not specified (Standard POSSM uses 1 per 50ms chunk)
        self.num_latents = getattr(config, 'num_latents', 1) 
        self.bin_size = getattr(config, 'bin_size', 50)
        self.embed_dim = config.embed_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.dropout = config.dropout
        
        # GQA / MQA support
        self.n_local_heads = self.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        
        # LEARNABLE QUERY: q in R^{1 x M} (or num_latents x embed_dim) 
        # This parameter replaces the projection of x into q.
        self.latent_query = nn.Parameter(torch.randn(self.num_latents, self.embed_dim))
        
        # Projections for Keys and Values (Input Spikes -> K, V)
        self.k_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=False)
        
        # Output projection for the latent representation z
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.embed_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.dropout = self.dropout
        
        # Initialize query projection for the learnable latents to match multi-head dimension
        self.q_proj = nn.Linear(self.embed_dim, self.num_attention_heads * self.head_dim, bias=False)
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, spike, offsets, spike_lengths, cos, sin):
        """
        Args:
            spike: Input spike tokens (batch, k_history, max_spikes_in_batch, embed_dim) -> Acts as Key/Value 
            offsets: Input spike offsets (batch, k_history, max_spikes_in_batch) 
            spike_lengths: Input spike lengths (batch, k_history)
            cos: RoPE cos for the spike tokens, shape (bin_size, head_dim)
            sin: RoPE sin for the spike tokens, shape (bin_size, head_dim)
        """
        batch_size, k_history, max_spikes_in_batch, _ = spike.shape
        
        # 1. Prepare Queries (Latents)
        # Expand learnable latent queries to match batch size
        # Shape: (Num_Latents, Dim)
        latents = self.latent_query
        xq = self.q_proj(latents).unsqueeze(0).unsqueeze(0).expand(batch_size, k_history, -1, -1) # (batch_size, k_history, num_latents, num_attention_heads * head_dim)
        
        # 2. Prepare Keys and Values (Spikes)
        xk = self.k_proj(spike) # (batch_size, k_history, max_spikes_in_batch, num_key_value_heads * head_dim)
        xv = self.v_proj(spike) # (batch_size, k_history, max_spikes_in_batch, num_key_value_heads * head_dim)
        
        # Reshape for Multi-Head Attention
        xq = xq.view(batch_size, k_history, self.num_latents, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, k_history, max_spikes_in_batch, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(batch_size, k_history, max_spikes_in_batch, self.n_local_kv_heads, self.head_dim)
        
        # 3. Apply RoPE to Spike Tokens (Keys)
        # The paper states spike tokens are encoded with RoPE.
        # Since latent queries are static/learned, we typically only apply RoPE to the Keys here 
        # to encode relative spike timing.
        xk = RotaryEmbedding.apply_rotary_pos_emb(xq, xk, cos, sin, offsets, is_decoder=False)

        # 4. Transpose for Attention
        xq = xq.transpose(2, 3) # (batch_size, k_history, num_attention_heads, num_latents, head_dim)
        xk = repeat_kv(xk, self.n_rep).transpose(2, 3) # (batch_size, k_history, num_key_value_heads, max_spikes_in_batch, head_dim)
        xv = repeat_kv(xv, self.n_rep).transpose(2, 3) # (batch_size, k_history, num_key_value_heads, max_spikes_in_batch, head_dim)
        
        # 5. Calculate Attention
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # (batch_size, k_history, num_attention_heads, num_latents, max_spikes_in_batch)

        # 去除padding的spikes
        # 步骤 A: 生成位置索引 [0, 1, ..., max_spikes-1]
        # shape: (1, 1, 1, 1, max_spikes_in_batch)
        position_ids = torch.arange(max_spikes_in_batch, device=spike.device).view(1, 1, 1, 1, -1)
        
        # 步骤 B: 调整 lengths 的形状以进行广播
        # shape: (batch_size, k_history, 1, 1, 1)
        # 我们需要在 batch 和 k_history 维度上区分长度，但在 head 和 latent 维度上共享
        lengths_expanded = spike_lengths.view(batch_size, k_history, 1, 1, 1)
        
        # 步骤 C: 生成布尔 Mask
        # 逻辑：如果当前位置索引 >= 真实长度(True)，则说明是 Padding (需要被 Mask 掉)
        # Result shape: (batch_size, k_history, 1, 1, max_spikes_in_batch)
        mask = position_ids >= lengths_expanded 
        
        # 步骤 D: 应用 Mask
        # 将 Mask 为 True (即 Padding 部分) 的位置设为负无穷
        scores = scores.masked_fill(mask, float('-inf'))
        
        scores = self.attn_dropout(scores)
        output = scores @ xv # (batch_size, k_history, num_attention_heads, num_latents, head_dim)

        # 6. Output Projection
        output = output.transpose(2, 3).reshape(batch_size, k_history, self.num_latents, -1) # (batch_size, k_history, num_latents, n_heads * head_dim)
        output = self.resid_dropout(self.o_proj(output)) # (batch_size, k_history, num_latents, embed_dim)
        
        # Return output
        return output