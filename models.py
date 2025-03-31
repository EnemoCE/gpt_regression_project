from copy import copy, deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
import types
from torch import optim
from torchdiffeq import odeint
from dataclasses import dataclass
from tqdm import tqdm
import math
import wandb


import numpy as np


from transformers import GPT2Model, GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from configurations import update_transform


import numbers
import torch.nn.init as init
from typing import Union, List
from torch import Size, Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'









class IteratedNewtonModel(nn.Module):
    def __init__(self, input_dim, num_iterations):
        super(IteratedNewtonModel, self).__init__()
        self.input_dim = input_dim
        self.num_iterations = num_iterations
        self.device = device
        self.S = None
        self.X = None

    def initialize(self, A):
        self.S = A.transpose(1,2) @ A 
        SS_T = self.S @ self.S.transpose(1, 2)
        eigenvalues = torch.linalg.eigvalsh(SS_T)
        lambda_max = torch.max(eigenvalues).item()
        alpha = 2 / lambda_max
        self.X = alpha * self.S

    def iterate(self, A):
        for step in range(self.num_iterations):
            self.X = 2 * self.X - self.X @ self.S @ self.X

    def forward(self, A, y):
        if self.X is None:
            self.initialize(A)
        self.iterate(A)
        self.w = (self.X @ A.transpose(1, 2) @ y.unsqueeze(2))
        return self.w.squeeze(2)



N_EPOCH = 10_000

N = 8
N_samples = 2048

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)




class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], eps: float = 1e-5, bias: bool = False) -> None:
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        var = input.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = input * torch.rsqrt(var)

        rmsnorm = self.weight * input_norm
        
        if self.bias is not None:
            rmsnorm = rmsnorm + self.bias

        return rmsnorm

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    #print(attn_bias.shape)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    #print(attn_mask.shape, 'attention_mask')
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
    
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
    
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    #print(attn_weight.shape, query.shape, key.shape, attn_bias.shape)
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.nan_to_num(attn_weight, nan=0.0)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = torch.einsum("...,i->...i", t.float(), freqs)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
    
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)
        self.rms_norm = RMSNorm(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k =self.rms_norm(q), self.rms_norm(k) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
#         y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=torch.ones((T,T),dtype=q.dtype,device=q.device))
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.rms_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.rms_norm(x))
        x = x + self.mlp(self.rms_norm(x))
        return x
    
@dataclass
class GPTConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 32
    y_n_positions: int = 25
    context_dim: int = 3

class CFMTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projections
        self.feature_proj = nn.Linear(config.y_n_positions, config.n_embd)
        self.x_t_proj = nn.Linear(1, config.n_embd)
        self.time_embed = TimestepEmbedder(config.n_embd)
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.head = nn.Linear(config.n_embd, 1)

    def forward(self, t, x_t, features):
        
        features_permuted = features.unsqueeze(-2) # (B*TC, 1, TC)
        features_emb = self.feature_proj(features_permuted) #(B*TC, 1, E)
        
        x_t_emb = self.x_t_proj(x_t).unsqueeze(-2)  # (B*TC, 1, E)
        t_emb = self.time_embed(t).unsqueeze(-2)     # (B*TC, 1, E)
        #print('x',x_t_emb.shape, 't',t.shape, 'f', features_emb.shape)
        sequence = torch.cat([t_emb, x_t_emb, features_emb], dim=-2)
        
        x = sequence
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x[:, t_emb.shape[-2]:t_emb.shape[-2] + x_t_emb.shape[-2], :])


@dataclass
class F_GPTConfig:
    n_layer: int = 4 
    n_head: int = 4
    n_embd: int = 32 # Should match ModifiedCFMTransformer input dim
    out_n_embd: int = 5

class FeatureProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
        self.time_embed = TimestepEmbedder(config.n_embd)
        self.hs_proj = nn.Linear(config.out_n_embd, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.head = nn.Linear(config.n_embd, 1)

    def forward(self, sequence):
        
        #t_emb = self.time_embed(t).unsqueeze(-2)     # (B*TC, 1, E)
        #print('x',x_t_emb.shape, 't',t.shape, 'f', features_emb.shape)
        
        
        x = self.hs_proj(sequence)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, Value projections for context
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Query projection for input x
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        # self.rotary = Rotary(self.head_dim) # Rotary usually applied to Q, K in self-attention
        # self.rms_norm = RMSNorm(self.head_dim) # Optional normalization

    def forward(self, x, context, attn_mask=None):
        B, T_q, C = x.size()
        B, T_kv, C_kv = context.size() # Assume C_kv == C for now

        q = self.c_q(x).view(B, T_q, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T_q, hs)
        k = self.c_k(context).view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T_kv, hs)
        v = self.c_v(context).view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T_kv, hs)
        #print(q.shape, k.shape, v.shape, 'c')
        # Optional QK normalization / Rotary embeddings can be added here if desired

        # Scaled Dot Product Attention
        # attn_mask should be (B, T_q, T_kv) or broadcastable
        y = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False) # is_causal=False for cross-attn

        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class AltBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_q = nn.LayerNorm(config.n_embd)
        #self.ln_kv = nn.LayerNorm(config.n_embd) # Assuming context has same dim
        self.rms_norm_kv = RMSNorm(config.n_embd)
        self.self_attn = CausalSelfAttention(config) # For interaction between t and z_t
        self.cross_attn = CrossAttention(config) # For interaction with processed features
        self.ln_mlp = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.rms_norm_pre_cross = RMSNorm(config.n_embd)
        self.rms_norm_pre_mlp = RMSNorm(config.n_embd)
        self.rms_norm_pre_self = RMSNorm(config.n_embd) # Added for self-attention

    def forward(self, x, context, cross_attn_mask=None):
        # x shape: (B_eff, 2, E)  -> [t_emb, z_emb]
        # context shape: (B_eff, TC, E) -> processed_hs for the relevant batch item
        # cross_attn_mask shape: (B_eff, 2, TC) -> allows attention only to <= i-1 features

        # Self-attention between t and z_t (causal mask within CausalSelfAttention not really needed for T=2, but harmless)
        # Using RMSNorm before attention as in the original Block
        print(torch.isnan(x).any().item(), 'pr_a')
        x = x + self.self_attn(self.rms_norm_pre_self(x))
        # Cross-attention: x attends to context
        # Using RMSNorm before attention
        print(torch.isnan(x).any().item(), 'a')
        x = x + self.cross_attn(self.rms_norm_pre_cross(x), self.rms_norm_kv(context), attn_mask=cross_attn_mask) # Using LayerNorm for context KV like original transformer
        #print(cross_attn_mask, 'cc')
        print(torch.isnan(x).any().item())
        # MLP
        # Using RMSNorm before MLP
        x = x + self.mlp(self.rms_norm_pre_mlp(x))
        return x


@dataclass
class M_GPTConfig:
    n_layer: int = 4 
    n_head: int = 4
    n_embd: int = 32 # Should match FeatureProcessor output dim
    out_n_embd: int = 5

class ModifiedCFMTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.z_h_proj = nn.Linear(config.out_n_embd, config.n_embd) # Project z_h
        self.time_embed = TimestepEmbedder(config.n_embd)

        self.blocks = nn.ModuleList([AltBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(config.n_embd, config.out_n_embd)

    def forward(self, t, z_h, processed_hs):
        # t: (B*TC,) - time steps
        # z_h: (B*TC, E) - interpolated hidden states (already projected if needed)
        # processed_hs: (B, TC, E) - pre-processed features

        B_eff = t.shape[0] # B*TC
        B, TC, E = processed_hs.shape

        t_emb = self.time_embed(t)     # (B_eff, E)
        z_emb = self.z_h_proj(z_h)   # (B_eff, E)

        # Combine t and z embeddings
        tz_seq = torch.stack([t_emb, z_emb], dim=1) # (B_eff, 2, E)

        # Get the corresponding full context sequence for each item in the batch
        b_indices = torch.arange(B, device=processed_hs.device).repeat_interleave(TC)
        context = processed_hs[b_indices] # (B_eff, TC, E)
        
        i_indices = torch.arange(TC, device=processed_hs.device).repeat(B)
        # i_indices: (B*TC,) - maps flattened index to sequence index (0 to TC-1)
        # Create the cross-attention mask
        # Mask shape (B_eff, 2, TC). Allows q (t,z) to attend to kv (context)
        n_head = self.config.n_head
        mask = torch.arange(TC, device=t.device).view(1, 1, 1, TC).expand(1, n_head , 2, TC) < i_indices.view(B_eff, 1, 1, 1).expand(B_eff, n_head, 2, 1)
        cross_attn_mask = torch.zeros_like(mask, dtype=tz_seq.dtype)
        cross_attn_mask.masked_fill_(~mask, -float('inf')) # Fill -inf where k >= i

        x = tz_seq
        for block in self.blocks:
            x = block(x, context, cross_attn_mask=cross_attn_mask)
            #print(torch.isnan(x).any().item())

        x = self.ln_f(x)

        # z_emb (index 1)
        v_h_pred = self.head(x[:, 1, :]) # (B_eff, E)

        return v_h_pred




class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, transform_params=None, auto_transform_params = None, 
                 model_variants = ['basic'], output_attentions=False):
        super(TransformerModel, self).__init__()
        self.configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.cfm_configuration = GPTConfig(y_n_positions=n_positions)
        self.modified_cfm_configuration = M_GPTConfig(out_n_embd=n_embd)
        self.feature_processor_config = F_GPTConfig(out_n_embd=n_embd)
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.transform_params = transform_params
        self.auto_transform_params = auto_transform_params
        self.model_variants = model_variants
        self.output_attentions = output_attentions
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(self.configuration)
        self._feature_processor = FeatureProcessor(self.feature_processor_config)
        self._cfm_read_out = CFMTransformer(self.cfm_configuration)
        self._alt_cfm_read_out = ModifiedCFMTransformer(self.modified_cfm_configuration)
        self.new_configuration = self.build_new_config()
        self.update_new_backbone()
        self._read_out = nn.Linear(n_embd, 1)
        self._read_out2 = nn.Linear(n_embd, 1)
        self._transmit = nn.Linear(n_embd, n_embd)


    def build_new_config(self):
        new_configuration = deepcopy(self.configuration)
        new_n_layer = self.configuration.n_layer

        if self.transform_params.duplicate_params:
            start, end, repeat = self.transform_params.duplicate_params
            new_n_layer = self.configuration.n_layer + (end + 1 - start) * (repeat - 1)
        elif self.transform_params.slice_params:
            start, end= self.transform_params.slice_params
            new_n_layer = end + 1 - start
        elif self.transform_params.average_params:
            new_configuration.average_params = self.transform_params.average_params

        new_configuration.n_layer = new_n_layer
        return new_configuration



    

    @staticmethod
    def average_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        

        if self.config.average_params:
            start, end= self.config.average_params
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0


        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if start <= i <= end:
                if i == start:
                    parallel_states = outputs[0].unsqueeze(0)
                else:
                    parallel_states = torch.cat((parallel_states, outputs[0].unsqueeze(0)), dim=0)
                if i == end:
                    hidden_states = parallel_states.mean(dim=0)
            else: 
                hidden_states = outputs[0]

            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def base_backbone_updater(self, layers):
        new_backbone = self._backbone.__class__(self.new_configuration)
        check_out = [id(ele) for ele in self._backbone.h.parameters()]
        with torch.no_grad():
            for paramA, paramB in zip(new_backbone.parameters(), self._backbone.parameters()):
                if id(paramB) not in check_out:
                    paramA.copy_(paramB)
        # with torch.no_grad():
        #     for i, layer in enumerate(layers):
        #         for paramA, paramB in zip(new_backbone.h[i].parameters(), layer.parameters()):
        #             paramA.copy_(paramB)
        
        new_backbone.h = nn.ModuleList(layers)
        return new_backbone
        

    def recompose(self, model_variant):
        if model_variant == 'modified':
            layers = list(self._backbone.h.children())
            transform_params = self.transform_params
            if transform_params:
                layers = transform_params.transform_func(layers)
            new_backbone = self.base_backbone_updater(layers)
        elif model_variant == 'full_backbone + no_final_layer_norm':
            new_backbone = deepcopy(self._backbone)
            new_backbone.ln_f = nn.Identity()
        else:
            return None
        return new_backbone.to(device)



    def update_new_backbone(self):
        self._h_new_backbone = dict()
        for ele in self.model_variants:
            backbone = self.recompose(ele)
            if self.transform_params.average_params and backbone is not None:
                backbone.forward = types.MethodType(self.average_forward, backbone)
            self._h_new_backbone[ele] = backbone

    def clear_readout2(self):
        self._read_out2 = self._read_out


    def _new_backbone(self, *args, model_variant=None, **kwargs):
        return self._h_new_backbone[model_variant](*args, **kwargs)


    def auto_recompose(self, hard_update=False):
        layers = list(self._backbone.h.children())
        if self.auto_transform_params:
            layers = self.auto_transform_params.auto_transform_func(layers)
        if hard_update:
            self._backbone = self.base_backbone_updater(layers).to(device)
            return
        self._backbone.h = nn.ModuleList(layers)
    
    
    @torch.no_grad()
    def collect_embedding(self, xs):
        #self.update_new_backbone()
        self.auto_recompose()
        #variant = [i for i in  range(t_v) if t_v[i] == "slice_layers"][0] + 1
        #self.transform_params, self.auto_transform_params = update_transform(self.transform_params, self.auto_transform_params, variant)
        #self.auto_recompose()
        return self._forward_base_post_eval_hidden(xs)


    @torch.no_grad()
    def _forward_base_post_eval(self, embeds, cfm=0):
        first_n_layers = self.transform_params.first_n_layers
        in_output = embeds.clone().detach()
        logits = None
        in_output = self._backbone(inputs_embeds=in_output, output_hidden_states=True).hidden_states[first_n_layers]
        if not cfm:
            with torch.set_grad_enabled(True):
                logits =  self._read_out2(in_output)
        else:
            logits =  self._read_out(in_output)
        return logits
    
    @torch.no_grad()
    def _forward_base_post_eval_hidden(self, xs, targets=None, base_model=True):
        embeds = self._read_in(xs)
        first_n_layers = self.transform_params.first_n_layers
        in_output = embeds.clone().detach()
        return self._backbone(inputs_embeds=in_output, output_hidden_states=True).hidden_states[first_n_layers]

    
    def _forward_modified(self, embeds):
        no_layernorm_full_backbone_copy = self.transform_params.no_layernorm_full_backbone_copy
        first_n_layers = self.transform_params.first_n_layers
        full_backbone_rnn_iters = self.transform_params.full_backbone_rnn_iters
        in_output = embeds.clone().detach()

        k = 1 if no_layernorm_full_backbone_copy else 0
        logits2 = None

        with torch.set_grad_enabled(self.transform_params.new_backbone_training):
            if not first_n_layers:
                first_n_layers =  self.new_configuration.n_layer * full_backbone_rnn_iters

            for i in range(full_backbone_rnn_iters):
                full_backbone_count =  self.new_configuration.n_layer * (i+1)
                if full_backbone_count > first_n_layers:
                    current_n_layers =  first_n_layers % (self.new_configuration.n_layer * i) if i else first_n_layers
                    in_output = self._new_backbone(inputs_embeds=in_output, model_variant = self.model_variants[k],
                                                    output_hidden_states=True).hidden_states[current_n_layers]
                    break
                in_output = self._new_backbone(inputs_embeds=in_output, model_variant = self.model_variants[k]).last_hidden_state
        
        if self.transform_params.cfm_loss[1] == 2:
            return in_output
        
        if self.transform_params.readout2_training:
            logits2 = self._read_out2(in_output)
        else:
            logits2 = self._read_out(in_output)
        
        return logits2

    def sample_conditional_pt(self, x0, x1, t, sigma):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t = t * x1 + (1 - t) * x0
        epsilon = torch.randn_like(x0)
        return mu_t + sigma * epsilon
        
    def compute_conditional_vector_field(self, x0, x1):
        return x1 - x0

    def cfm_loss(self, logits, targets):

        B, TC = logits.shape       
        x0 = logits.view(B*TC, 1)     # B, TC ->  B*TC, 1
        features = logits.view(B, 1, TC).expand(B, TC, TC)  # B, TC -> B, TC, TC
        mask = torch.tril(torch.ones(TC, TC), diagonal=-1)  
        mask = mask.to(features.device)
        #mask = mask.unsqueeze(0)  # (1, TC, TC)  not needed due to automatic broadcasting
        features = features * mask  # (B, TC, TC')
        features = features.view(B*TC, -1) #(B*TC, TC')
        t = torch.rand(B*TC).type_as(x0)
        #t = torch.zeros(B*TC).type_as(x0)
        #print('x0', x0.shape, 't', t.shape, 'f', features.shape)
        xt = self.sample_conditional_pt(x0, targets.view(B*TC, 1), t, sigma=0.0)
        ut = self.compute_conditional_vector_field(x0, targets.view(B*TC, 1))
        vt = self._cfm_read_out(t, xt, features)
        vt = vt[:,0,:].view(B*TC)
        ut = ut[:,0]
        loss = F.mse_loss(vt, ut)
        return vt, loss
    
    @torch.no_grad()
    def h_inv(self, y_scalar):
        # y_scalar shape: (N, 1) or (N,)
        # Returns h_approx shape: (N, C) where C = n_embd
        self._read_out2.eval() 

        y_scalar = y_scalar.view(-1, 1) # Ensure shape (N, 1)
        W = self._read_out2.weight # Shape: (1, C)
        b = self._read_out2.bias   # Shape: (1,)
        WWt = W @ W.T
        # Add small epsilon for numerical stability if WWt can be zero
        WWt_inv = 1.0 / WWt.clamp(min=1e-6) # Shape: (1, 1)
        h_approx = (y_scalar - b) * WWt_inv @ W 
        return h_approx.detach() # Shape: (N, C)
    
    def alt_cfm_loss(self, hs_pred, targets):
        B, T, C = hs_pred.shape 
        h_targets =  self.h_inv(targets.view(B*T, 1)) # (B*T, C)
        #print(self._read_out2(h_targets)[0:5], targets.view(B*T, 1)[0:5])
        processed_features_hs = self._feature_processor(hs_pred) # (B, T, E)
        #processed_features_hs = self._feature_processor(h_targets.view(B, T, C)) # (B, T, E)
        #x0 = logits.view(B*T)
        t = torch.rand(B*T).type_as(hs_pred)
        z_h = self.sample_conditional_pt(hs_pred.view(B*T, C), h_targets, t, sigma=0.0) #(B*T, C)
        h_ut = self.compute_conditional_vector_field(hs_pred.view(B*T, C), h_targets)
        #ut = self.compute_conditional_vector_field(x0, targets.view(B*T))
        print(torch.isnan(hs_pred).any().item(), 'hs_pred')
        h_vt = self._alt_cfm_read_out(t, z_h, processed_features_hs) # (B*T, C)
        print(1)
        #print(h_vt)
        vt = h_vt.detach() #(B*T, C)
        #loss_y = F.mse_loss(vt, ut)  #training only _read_out
        loss_h = F.mse_loss(h_vt, h_ut)
        #print(loss_h.item())
        #loss = loss_h + loss_y
        return vt, loss_h


    # cfm_loss 0,0 -> none    cfm_loss 1,0 -> modified     cfm_loss 2,0 -> base, modified    cfm_loss 2,1 -> base, modified + aditional train,  2, 2 alt_training
    def do_cfm(self, cfm_loss, base_model):
        do_cfm_states = {
        0:[False, False],
        1:[False, True],
        2:[True, True],
        }
        model_var = 0 if base_model else 1
        return do_cfm_states[cfm_loss[0]][model_var]



    def forward(self, xs, targets=None, base_model=True):

        embeds = self._read_in(xs)
        output = self._backbone(inputs_embeds=embeds, output_attentions=self.output_attentions, output_hidden_states=True)
        logits = None
        cfm_loss = self.transform_params.cfm_loss

        if base_model and not self.transform_params.post_eval:
            logits = self._read_out(output.last_hidden_state)
        elif base_model or cfm_loss[1] == 1:
            logits = self._forward_base_post_eval(embeds, self.cfm_loss[1])
        else:
            logits = self._forward_modified(embeds)
        
        if targets is None:
            logits = logits.detach()
            loss = None
            if self.do_cfm(cfm_loss, base_model) and cfm_loss[1] == 2:
                hs_pred = logits[:,::2,:]
                return hs_pred, loss
            logits = logits[:, ::2, 0]
        else:
            targets = targets[:, ::2, 0]
            if self.do_cfm(cfm_loss, base_model):
                if cfm_loss[1] == 2:
                    hs_pred = logits[:,::2,:]
                    logits, loss = self.alt_cfm_loss(hs_pred, targets)
                else:
                    logits = logits[:, ::2, 0]
                    logits, loss = self.cfm_loss(logits, targets)
                return logits, loss
            logits = logits[:, ::2, 0]
            loss = F.mse_loss(logits, targets)
        return logits, loss




def build_model(conf):
    if conf.model.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.model.n_dims,
            n_positions=conf.model.n_positions,
            n_embd=conf.model.n_embd,
            n_layer=conf.model.n_layer,
            n_head=conf.model.n_head,
            model_variants = conf.experiment_conf.transform_conf.model_variants,
            transform_params = conf.experiment_conf.transform_conf,
            auto_transform_params = conf.experiment_conf.auto_transform_conf
        )
    else:
        raise NotImplementedError
    return model
