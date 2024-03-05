import torch
import torch.nn as nn
from .window_blocks import WindowAttentionGlobal, window
from .cross_attn import CrossAttention
from .backbones import Feature_Extractor
from .raf_db_loader import RAF_DB_Loader
from .vit_model import PatchEmbed, VisionTransformer
from timm.models.layers import DropPath
from .lbp_extractor import LBP_Extractor
#from .lbplib import LBP_Extractor



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

def lm_to_q(lm):
    B, C, H, W = lm.shape
    return lm.permute(0, 2, 3, 1).reshape(B, H*W, C)

def _channel_last_to_q(gcip):
    B, H, W, C = gcip.shape
    return gcip.reshape(B, H*W, C)

def reverse_BHWC(x, shortcut):
    B, H, W, C = shortcut.shape
    return x.view(B, H, W, -1)

def shape(xs):
    for x in xs:
        print(x.shape)

class HFFT_LBP(nn.Module):
    def __init__(self, window_size=[28, 14, 7], dims=[64, 128, 256],
                 num_heads = [2, 4, 8], LBP_gray = False):
        super().__init__()
        self.window_sizes = window_size
        self.dims = dims
        self.N = [w * w for w in window_size]
        self.dim_heads = [int(torch.div(dim, num_head).item()) for dim, num_head in zip(dims, num_heads)]  
        self.num_heads = num_heads
        self.nfs = len(dims) # num of feature scales

        self.fea_ext = Feature_Extractor()
        self.windows = nn.ModuleList([window(self.window_sizes[i], dims[i]) for i in range(self.nfs)])
        dpr = [x.item() for x in torch.linspace(0, 0.5, 5)]
        # Global-local fusion
        self.attn_GCIP = nn.ModuleList([WindowAttentionGlobal(dims[i], num_heads[i],window_size[i]) for i in range(self.nfs)])
        self.ffn_GCIP = nn.ModuleList([FFN(dims[i],window_size[i],layer_scale=1e-5, drop_path=dpr[i]) for i in range(self.nfs)])
        # landmark-texture fusion
        self.attn_GCIA = nn.ModuleList([CrossAttention(dims[i], num_heads[i]) for i in range(self.nfs)])
        self.ffn_GCIA = nn.ModuleList([CrossFFN(dims[i],window_size[i],layer_scale=1e-5, drop_path=dpr[i]) for i in range(self.nfs)])
        # enhancement fusion
        self.lbp_ext = LBP_Extractor(dims, gray=LBP_gray)
        self.attn_EHCA = nn.ModuleList([CrossAttention(dims[i], num_heads[i]) for i in range(self.nfs)])
        self.ffn_EHCA = nn.ModuleList([CrossFFN(dims[i],window_size[i],layer_scale=1e-5, drop_path=dpr[i]) for i in range(self.nfs)])
        
        # ViT
        self.embed_q = nn.Sequential(nn.Conv2d(dims[0], 768, kernel_size=3, stride=2, padding=1),
                             nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1))
        self.embed_k = nn.Sequential(nn.Conv2d(dims[1], 768, kernel_size=3, stride=2, padding=1))
        self.embed_v = PatchEmbed(img_size=14, patch_size=14, in_c=256, embed_dim=768)
        self.VIT = VisionTransformer(depth=2, embed_dim=768)
        
    def forward(self, x):
        lms, irs = self.fea_ext(x)
        # GCIP
        # split irs to windows, then use the original shortcut as query
        window_splits = [list(self.windows[i](irs[i])) for i in range(self.nfs)]
        x_windows, shortcuts = [c[0] for c in window_splits], [c[1] for c in window_splits]
        q_irgs = [_to_query(shortcuts[i], self.N[i], self.num_heads[i], self.dim_heads[i]) for i in range(self.nfs)] # irs global query
        gcip_outs = [self.attn_GCIP[i](x_windows[i], q_irgs[i]) for i in range(self.nfs)]
        ffn_gcip_outs = [self.ffn_GCIP[i](gcip_outs[i], shortcuts[i]) for i in range(self.nfs)]
        
        # FSFP
        # use gcip_outs as query, lms as kv
        q_lms = [lm_to_q(lm) for lm in lms] 
        q_gcips = [_channel_last_to_q(gcip) for gcip in ffn_gcip_outs]
        gcia_outs = [self.attn_GCIA[i](q_gcips[i], q_lms[i], ) for i in range(self.nfs)] # B, H*W, C
        ffn_gcia_outs = [self.ffn_GCIA[i](gcia_outs[i], ffn_gcip_outs[i]) for i in range(self.nfs)]
        
        # EHCA
        lbps = self.lbp_ext(x)
        q_gcias = [_channel_last_to_q(gcia) for gcia in ffn_gcia_outs]
        q_lbps = [_channel_last_to_q(lbp.permute(0, 3, 1, 2)) for lbp in lbps]
        ehca_outs = [self.attn_EHCA[i](q_gcias[i], q_lbps[i]) for i in range(self.nfs)]
        ffn_ehca_outs = [self.ffn_EHCA[i](ehca_outs[i], ffn_gcia_outs[i]) for i in range(self.nfs)]

        outs = [_to_channel_first(o) for o in ffn_ehca_outs]
        o1, o2, o3 = self.embed_q(outs[0]).flatten(2).transpose(1, 2), self.embed_k(outs[1]).flatten(2).transpose(1, 2), self.embed_v(outs[2])
        o = torch.cat([o1, o2, o3], dim=1)
        out = self.VIT(o)
        return out
    
        
class FFN(nn.Module):
    def __init__(self, dim, window_size, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0.,layer_scale=None):
        super().__init__()
        if layer_scale is not None and type(layer_scale) in [int, float]: # gammas
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

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
    
class CrossFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0., layer_scale=None):
        super().__init__()
        self.mlp = MLP(dim, int(dim*mlp_ratio), act_layer = act_layer, drop=drop)
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, shortcut):
        x = reverse_BHWC(x, shortcut)
        x = shortcut + self.drop_path(self.mlp(self.norm(x)))
        return x
    