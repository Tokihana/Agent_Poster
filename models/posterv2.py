from msilib.schema import Patch
from telnetlib import KERMIT
from xml.etree.ElementInclude import include
import torch
import torch.nn as nn
from torch.nn import functional as F
from .backbones import Feature_Extractor
from .raf_db_loader import RAF_DB_Loader
from .window_blocks import window, WindowAttentionGlobal
from .clean_vit_model import VisionTransformer, PatchEmbed
from timm.models.layers import DropPath
from .gc_vit import GCViTBlock

def window_reverse(windows, window_size, H, W, h_w, w_w):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, h_w, w_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def _to_query(x, N, num_heads, dim_head):
    B = x.shape[0]
    x = x.reshape(B, 1, N, num_heads, dim_head).permute(0, 1, 3, 2, 4)
    return x

def _to_channel_first(x):
    return x.permute(0, 3, 1, 2)

def _to_channel_last(x):
    return x.permute(0, 2, 3, 1)

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        return self.model(x)
    
class FFN(nn.Module):
    def __init__(self, dim, window_size, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0.,layer_scale=None):
        super().__init__()
        if layer_scale is not None and type(layer_scale) in [int, float]: # gammas
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1=1.0
            self.gamma2=1.0

        self.window_size = window_size
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer= act_layer, drop=drop)
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, attn_windows, shortcut):
        B, H, W, C = shortcut.shape
        h_w = int(torch.div(H, self.window_size).item())
        w_w = int(torch.div(W, self.window_size).item())
        x = window_reverse(attn_windows, self.window_size, H, W, h_w, w_w)
        x = shortcut + self.drop_path(self.gamma1 * x) # first res 
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm(x))) # second res
        return x       
    
class PosterV2(nn.Module):
    def __init__(self, dims=[64, 128, 256], window_sizes=[28, 14, 7], num_heads=[2, 4, 8], deepmad=False, embed_dim=768, use_blocks = True,pool_embed=False):
        super().__init__()
        self.feature_scales = len(dims)
        self.N = [w * w for w in window_sizes]
        self.dims = dims
        self.window_sizes = window_sizes
        self.num_heads = num_heads
        self.dim_head = [int(torch.div(dim, num_head).item()) for num_head, dim in zip(num_heads, dims)]
        # Global lm attention
        self.feature_extractor = Feature_Extractor(use_blocks=use_blocks)
        self.use_blocks = use_blocks
        dpr = [x.item() for x in torch.linspace(0, 0.5, 5)]
        if use_blocks:
            self.blocks = nn.ModuleList([GCViTBlock(dims[i], window_sizes[i]*2, num_heads[i],window_sizes[i]) for i in range(self.feature_scales)])
        else:
            self.windows = nn.ModuleList([window(window_sizes[i], dims[i]) for i in range(self.feature_scales)])
            self.attns = nn.ModuleList([WindowAttentionGlobal(dims[i], num_heads[i], window_sizes[i]) for i in range(self.feature_scales)])
            self.ffns = nn.ModuleList([FFN(dims[i], window_sizes[i], layer_scale=1e-5, drop_path=dpr[i]) for i in range(self.feature_scales)])
        

        # ViT
        if pool_embed:
            self.embed_q = nn.Sequential(nn.Conv2d(dims[0], 768, kernel_size=3, stride = 2, padding=1), nn.AdaptiveAvgPool2d(window_sizes[2]))
            self.embed_k = nn.Sequential(nn.Conv2d(dims[1], 768, kernel_size=3, stride = 2, padding=1), nn.AdaptiveAvgPool2d(window_sizes[2]))
            self.embed_v = nn.Sequential(nn.Conv2d(dims[2], 768, kernel_size=3, padding=1), nn.AdaptiveAvgPool2d(window_sizes[2]))
        else:
            self.embed_q = nn.Sequential(nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1),
                             nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1),
                             nn.Conv2d(dims[2], 768, kernel_size=3, stride=2, padding=1),)           # 这样可能更好一些，且更符合代码图示
            self.embed_k = nn.Sequential(nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(dims[2], 768, kernel_size=3, stride=2, padding=1),)
            self.embed_v = nn.Sequential(nn.Conv2d(dims[2], 768, kernel_size=3, stride=2, padding=1))
        self.VIT = VisionTransformer(depth=2, embed_dim=embed_dim)
        
    def forward(self, x):
        lms, irs = self.feature_extractor(x)
        #print_shapes(lms)
        #print_shapes(irs)
        qs = [_to_query(_to_channel_last(lms[i]), self.N[i], self.num_heads[i], self.dim_head[i]) for i in range(self.feature_scales)]
        if self.use_blocks:
            attn_outs = [self.blocks[i](_to_channel_last(irs[i]), qs[i]) for i in range(self.feature_scales)]
            outs = [_to_channel_first(attn_out) for attn_out in attn_outs]
        else:
            window_splits = [list(self.windows[i](irs[i])) for i in range(self.feature_scales)]
            x_windows, shortcuts = [c[0] for c in window_splits], [c[1] for c in window_splits]
            attn_outs = [self.attns[i](x_windows[i], qs[i]) for i in range(self.feature_scales)]
        
            ffn_outs = [self.ffns[i](attn_outs[i], shortcuts[i]) for i in range(self.feature_scales)]
            outs = [_to_channel_first(o) for o in ffn_outs]
        
        #print_shapes(outs)

        o1, o2, o3 = self.embed_q(outs[0]).flatten(2).transpose(1, 2), self.embed_k(outs[1]).flatten(2).transpose(1, 2), self.embed_v(outs[2]).flatten(2).transpose(1,2)
        #print(o1.shape, o2.shape, o3.shape)
        o = torch.cat([o1, o2, o3], dim=1)
        out = self.VIT(o)
        return out
        
def print_shapes(x):
    print([e.shape for e in x])