import torch
import torch.nn as nn
from Cross_Attention import POSSMCrossAttention
from RoPE import RotaryEmbedding

from Config import my_POSSMConfig
config = my_POSSMConfig()

import json
meta_data = json.load(open("processed_data/meta_data.json", "r"))
max_time_length = meta_data["max_time_length"]

from Dataloader import get_dataloader    
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

    def forward(self, spike, spike_lengths):
        # spike: (batch, k_history, max_spikes_in_batch, 2)
        # spike_lengths: (batch, k_history)
        channels, offsets = spike[..., 0], spike[..., 1]
        emb = self.emb(channels) # (batch, k_history, max_spikes_in_batch, embed_dim)
        z = self.Cross_Attention(emb, offsets, spike_lengths, self.freqs_cos, self.freqs_sin) # (batch_size, k_history, num_latents, embed_dim)
        h, _ = self.gru(z) # h: (batch_size, k_history, hidden_dim)
        vel_pred = self.output_decoder(h, self.freqs_cos, self.freqs_sin) # (batch_size, bin_size, 2)
        return vel_pred