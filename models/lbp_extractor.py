from .lbcnn_model import BlockLBP
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from .ir50 import Backbone
from models.iresnet import iresnet18
import torchvision.transforms as transforms
from .fast_glcm_torch import fast_glcm_contrast


class LBP_Extractor(nn.Module):
    def __init__(self, dims = [64, 128, 256], gray = False):
        super().__init__()
        self.dims = dims
        self.gray = gray
        if gray:
            self.gray_trans = transforms.Grayscale()
        self.lbp_ext = BlockLBP(3, 8)
        self.ir18 = iresnet18(False) # 50 layers, drop = 0.0, mode = 'ir'
        ir_checkpoints = torch.load('./models/pretrain/ir18.pth', map_location = lambda storage, loc:storage)
        self.ir18.load_state_dict(ir_checkpoints)
        self.convs = nn.ModuleList(nn.Conv2d(in_channels=dims[i], out_channels=dims[i], kernel_size=3, padding=1, stride=2) for i in range(len(dims)))
        
        
    def forward(self, x):
        #contrast = fast_glcm_contrast(x, vmin=-2.2, vmax=2.7, angles=torch.tensor([0., 45., 90., 135.]))
        if self.gray:
            x = self.gray_trans.forward(x).repeat(1, 3, 1, 1)
        lbp = self.lbp_ext(x)
        #lbp = lbp + contrast
        x_lbps = list(self.ir18(lbp))
        x_lbps = [self.convs[i](x_lbps[i]) for i in range(len(self.dims))]
        return x_lbps