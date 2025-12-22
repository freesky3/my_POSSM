import torch
import torch.nn as nn
from Cross_Attention import POSSMCrossAttention
from RoPE import RotaryEmbedding

from Config import my_POSSMConfig
config = my_POSSMConfig()

import json
meta_data = json.load(open("processed_data/meta_data.json", "r"))
max_time_length = meta_data["max_time_length"]

from Dataloder import get_dataloader    
from GRU import POSSM_Backbone_GRU
from Output_Decoder import POSSMOutputDecoder


class my_POSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Spike Tokenization
        self.emb = nn.Embedding(num_embeddings=meta_data["num_channel"], embedding_dim=config.embed_dim)
        self.Cross_Attention = POSSMCrossAttention(config)
        self.gru = POSSM_Backbone_GRU(config.num_latents * config.embed_dim, config.gru_hidden_size, config.gru_num_layers, config.dropout)
        freqs_cos, freqs_sin = RotaryEmbedding.precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.bin_size, config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
        self.output_decoder = POSSMOutputDecoder(config)

    def forward(self, spike, mask_spike, lengths_spike):
        # spike: (batch, max_bin, max_token, 2)
        # mask_spike: (batch, max_bin, max_token)
        # channels: (batch, max_bin, max_token)
        # offsets: (batch, max_bin, max_token)
        channels, offsets = spike[..., 0], spike[..., 1]
        emb = self.emb(channels) # (batch, max_bin, max_token, embed_dim)
        z = self.Cross_Attention(emb, offsets, mask_spike, self.freqs_cos, self.freqs_sin) # (batch_size, max_bin, num_latents, embed_dim)
        h, _ = self.gru(z, lengths_spike) # h: (batch_size, max_bin, hidden_dim)
        vel_pred = self.output_decoder(h, self.freqs_cos, self.freqs_sin) # (batch_size, max_bin * bin_size, 2)
        vel_pred = vel_pred[:, :max_time_length, :] # (batch_size, max_time_length, 2)
        return vel_pred