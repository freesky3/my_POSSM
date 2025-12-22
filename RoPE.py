import torch
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

# shape of k,q: (Batch_Size, Num_Heads, Seq_Len, Head_Dim)
class RotaryEmbedding:
    """Implementation of the RoPE computation module"""
    @staticmethod
    def precompute_freqs_cis(dim: int, bin_size: int, theta: float = 1e6):
        """
        预计算旋转位置编码（RoPE）的频率张量。

        Args:
            dim (int): 注意力头的维度 (head_dim)。
                    通常等于 hidden_size // num_heads。
                    必须是偶数，因为 RoPE 是两两一组旋转的。
            bin_size (int): 预计算的最大 offset 长度。
            theta (float, optional): RoPE 计算周期的基频参数。

        Returns:
            torch.Tensor: 形状为 (bin_size, dim) 的张量，包含预计算的旋转因子。
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(bin_size, device=freqs.device)
        freqs = torch.outer(t, freqs).float() # tensor shape: [bin_size, dim/2]
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return torch.cat([freqs_cos, freqs_cos], dim=-1), torch.cat([freqs_sin, freqs_sin], dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, offset, is_decoder):
        """
        将旋转位置编码 (RoPE) 应用于 Query 和 Key 向量。

        Args:
            q (torch.Tensor): Query 状态张量。(只在is_decoder=True时使用)
                形状为 [batch_size, max_bin, bin_size, num_heads, head_dim]
            k (torch.Tensor): Key 状态张量。
                形状为 [batch_size, max_bin, max_token, num_key_value_heads, head_dim]
            cos (torch.Tensor): 预计算的 Cosine 值。
                形状应为 [bin_size, head_dim] 经过 boardcast 后能与 q, k 广播的形状相同
            sin (torch.Tensor): 预计算的 Sine 值。
                形状应为 [bin_size, head_dim] 经过 boardcast 后能与 q, k 广播的形状相同
            offset (batch_size, max_bin, max_token): 需要旋转位置编码向量的相对位置。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                包含旋转后的 (q_embed, k_embed) 的元组。
                q_embed: 形状为 [batch_size, max_bin, num_latents, num_heads, head_dim]
                k_embed: 形状为 [batch_size, max_bin, max_token, num_key_value_heads, head_dim]
        """
        def rotate_half(x):
            """
            将输入张量的最后一维分为两半，交换顺序并对前半部分取反。
            对应旋转矩阵：[x, y] -> [cos*x - sin*y, sin*x + cos*y]
            """
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        
        if is_decoder:
            bin_size, head_dim = cos.shape
            cos = cos.view(1, 1, bin_size, 1, head_dim)
            sin = sin.view(1, 1, bin_size, 1, head_dim)
            q = (q * cos) + (rotate_half(q) * sin)
            return q
        else:
            cos, sin = cos[offset].unsqueeze(-2), sin[offset].unsqueeze(-2) # cos, sin: [batch_size, max_bin, max_token, 1, head_dim]
            k = (k * cos) + (rotate_half(k) * sin)
            return k