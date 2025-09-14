from transformers import PretrainedConfig

class Config(PretrainedConfig):
    model_type = "small_model"
    
    def __init__(self,
                hidden_size = 512,
                num_attention_heads = 16,
                num_key_value_heads = 8,
                flash_attn = True,
                attention_bias = False,
                max_seq_len = 512,
                intermediate_size = None,
                mlp_bias = False,
                vocab_size = 6400,
                n_layers = 8,
                dropout = 0.0,
                multiple_of = 64,
                rms_norm_eps = 1e-5,
                pad_token_id = 3,
                **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.multiple_of = multiple_of
        self.pad_token_id = pad_token_id
        super().__init__(**kwargs)