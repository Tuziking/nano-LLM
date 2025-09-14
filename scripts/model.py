# model.py (片段)
import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional
from config import Config


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 可学习参数
        self.eps = eps # 防止除以0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, hidden_size]
        hidden_states = x.float()
        x_norm = hidden_states.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() # [batch_size, seq_len, 1] 
        return (hidden_states / x_norm).to(x.dtype) * self.weight
    
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)  # 分为前半和后半
    return torch.cat((-x2, x1), dim=-1)  # 交换并取负
    

def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
   
    q_embed = (q*cos) + (rotate_half(q)*sin)
    k_embed = (k*cos) + (rotate_half(k)*sin)
    
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
        
    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)
    
# 将 Key（K）或 Value（V）张量的注意力头（heads）重复多次，以匹配 Query（Q）的头数
def repeat_kv(hidden_states, n_rep):
    
    batch_size, seq_length, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch_size, seq_length, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch_size, seq_length, num_key_value_heads * n_rep, head_dim)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout   
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads # 注意力头的数量
        self.head_dim = self.hidden_size // self.num_heads # 每个头的维度
        self.num_key_value_heads = config.num_key_value_heads # kv头的数量
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # 
        self.k_cache, self.v_cache = None, None
        self.is_causal = True
        self.flash_attn = self.config.flash_attn

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads  * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads  * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.residual_dropout = nn.Dropout(self.dropout)
        self.attention_dropout = nn.Dropout(self.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, hidden_states, use_kv_cache=False):
        # hidden_states: [batch_size, seq_len, hidden_size]
        batch_size, seq_len = hidden_states.shape[:2]
        if use_kv_cache and not self.training:
            if self.k_cache is None or self.k_cache.shape[1] != seq_len-1:
                q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
            else:
                token = hidden_states[:, -1:, :]
                q = torch.cat((torch.zeros_like(hidden_states[:, :-1, :]), self.q_proj(token)), dim=1)
                k = torch.cat((self.k_cache, self.k_proj(token)), dim=1)
                v = torch.cat((self.v_cache, self.v_proj(token)), dim=1)
            self.k_cache, self.v_cache = k, v

        else:
            q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        # (batch_size, seq_len, num_heads * head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        q,k = self.rotary_emb(q, k) # 应用旋转位置编码

        k = repeat_kv(k, self.num_key_value_groups) # 重复kv以匹配q的头数
        v = repeat_kv(v, self.num_key_value_groups) # 重复kv以匹配q的头数

        # batch_size, seq_len, num_heads, head_dim => batch_size, num_heads, seq_len, head_dim
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) 

        if self.flash_attn:
        
            # q*k转置，（batch_size, self.num_heads, seq_len, self.head_dim）* (batch_size, self.num_heads, self.head_dim，seq_len) = （batch_size, self.num_heads, head_dim，seq_len, head_dim，seq_len）
            # q*k/sqrt(self.head_dim)*v  （b, self.num_heads, s, s）* (b, self.num_heads, s, self.head_dim) = b, self.num_heads, s, self.head_dim
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                    dropout_p=self.dropout if self.training else 0.0, 
                                                    is_causal=self.is_causal) 
        else:
            # 使用掩码来遮盖未来token
            mask = torch.full((1, 1, self.config.max_seq_len, self.config.max_seq_len), float("-inf"))  # 初始化掩码
            mask = torch.triu(mask, diagonal=1)  # 生成上三角掩码
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力分数
            mask = torch.full(
                (1, 1, seq_len, seq_len),
                float("-inf"),
                device=q.device
            ).triu(diagonal=1)
            scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)  # 计算 softmax
            scores = self.attention_dropout(scores)  # 应用注意力 dropout
            output = torch.matmul(scores, v)  # 计算输出

        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, num_heads * head_dim)

        output = self.o_proj(output) # 线性变换
        output = self.residual_dropout(output)
        return output
    

class FFN(nn.Module):
    def __init__(self, hidden_size:int, intermediate_size: int, multiple_of: int, dropout: float):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size # "Attention Is All You Need" 论文中的经典设置。
            intermediate_size = int(2 * intermediate_size / 3) # 缩放因子,PaLM 模型中被提出和验证，为了在保持性能的同时减少参数量和计算量。
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of) # 向上取整到最近的 multiple_of 的倍数，可以更有效地利用 GPU 的张量核心（Tensor Cores），从而加速计算。
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False) # "值" (value) 的投影
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False) # "下采样" (down-projection) 层。
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False) # "门" (gate) 的投影
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: Config):
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention = Attention(config)

        self.attention_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.ffn = FFN(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )
    
    # Norm -> Attention -> Add -> Norm -> FFN -> Add
    def forward(self, hidden_states, use_kv_cache=False):
        # Norm -> Attention -> Add
        hidden_states = self.attention_norm(hidden_states)
        attention_states = self.attention(hidden_states, use_kv_cache=use_kv_cache)
        hidden_states = hidden_states + attention_states
        
        # Norm -> FFN -> Add
        ffn_norm_states = self.ffn_norm(hidden_states)
        ffn_states = self.ffn(ffn_norm_states)
        output_states = hidden_states + ffn_states
        return output_states
    
class nanoLLM(PreTrainedModel):
    config_class = Config
    last_loss: Optional[torch.Tensor]

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size) # 词嵌入层
        self.layers = nn.ModuleList([TransformerBlock(i, config) for i in range(config.n_layers)]) # Transformer 层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 最后的 RMSNorm 层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # 语言模型头+
        self.dropout = nn.Dropout(config.dropout)
        
        self.token_embeddings.weight = self.lm_head.weight # 词嵌入和输出层共享权重

        self.apply(self._init_weights) # 初始化权重


        # 特殊的权重初始化,确保在训练之初，流经残差连接的信号不会爆炸(LLM常用的初始化方法)
        for name, parameter in self.named_parameters():
            if name.endswith('w3.weight') or name.endswith('o_proj.weight'):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))


        self.loss = None
        self.OUT = CausalLMOutputWithPast()
        self._no_split_modules = [name for name, _ in self.named_modules()] # 用于分布式训练时不拆分的模块列表


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, input_ids, labels, use_kv_cache=False):
    
        hidden_states = self.token_embeddings(input_ids) 
        hidden_states = self.dropout(hidden_states)  
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, use_kv_cache=use_kv_cache)  

        hidden_states = self.norm(hidden_states) 

        if labels is not None:
            logits = self.lm_head(hidden_states)  
            pad_id = self.config.pad_token_id if hasattr(self.config, "pad_token_id") else self.tokenizer.pad_token_id
            self.loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=pad_id
            )
        else:
            logits = self.lm_head(hidden_states[:, [-1], :])  
            self.loss = None  

        return CausalLMOutputWithPast(self.loss, logits)
    
    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1.,
                 use_kv_cache=True):
        
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        s = input_ids.shape[1]
        while input_ids.shape[1] < max_new_tokens - 1:  
            inference_res = self(input_ids, labels=None, use_kv_cache=use_kv_cache)
            logits = inference_res.logits 
            logits = logits[:, -1, :] 

            for token in set(input_ids.tolist()[0]):  
                logits[:, token] /= repetition_penalty

            if temperature == 0.0: 
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature  
                if top_k is not None:  
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf') 

                probs = F.softmax(logits, dim=-1)  
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)  

            if idx_next == eos:  
                break

            input_ids = torch.cat((input_ids, idx_next), dim=1)  
            if stream:  
                yield input_ids[:, s:]  

        if not stream:  
            yield input_ids[:, s:]  
