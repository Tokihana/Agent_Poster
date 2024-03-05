import torch
import torch.nn as nn
from models.vit_model import PatchEmbed
from timm.models.layers import trunc_normal_

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim// num_heads
        self.scale = head_dim ** -0.5
        
        self.kv = nn.Linear(dim, dim * 2, bias = qkv_bias) # * 2 only because q_c has given
        self.q = nn.Linear(dim, dim, bias = qkv_bias) # q attn map
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, q_c):
        B, N, C = x.shape
        head_dim = torch.div(C, self.num_heads, rounding_mode = 'floor')
        
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4) # (kv, B, num_heads, B, head_dim)
        k, v = kv[0], kv[1]
        q = self.q(x).reshape(B, N, 1, self.num_heads, head_dim).permute(2, 0, 3, 1, 4) # (q, B, num_heads, B, head_dim)
        q = q*self.scale
        
        attn = q@k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x