from transformers import PretrainedConfig

class my_POSSMConfig(PretrainedConfig):
    model_type = "my_POSSM"

    def __init__(
            self,
            time_lag =80, # kinematics data lag 80ms relative to neural data
            bin_size = 50,
            num_latents = 1, # number of latent queries

            # general model
            dropout: float = 0.0,
            hidden_act: str = 'silu', # core component of SwiGLU
            embed_dim: int = 64, # the dimension of word'set embedding vector
            hidden_size: int = 128,
            
            intermediate_size: int = None,
            vocab_size: int = 182,

            # attention
            num_learable_q = 1, 
            num_attention_heads: int = 8,  
            num_key_value_heads: int = 2,
            flash_attn: bool = False,

            # gru
            gru_hidden_size: int = 512,
            gru_num_layers: int = 1,
            
            # output decoder
            k_history: int = 3,

            # others
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            **kwargs
    ):
        super().__init__(
            **kwargs
            )
        self.time_lag = time_lag
        self.bin_size = bin_size
        self.num_latents = num_latents

        self.dropout = dropout
        self.hidden_act = hidden_act
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        
        self.num_learable_q = num_learable_q
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn

        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        self.k_history = k_history

        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
