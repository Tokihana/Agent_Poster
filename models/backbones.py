from re import M
from turtle import st
import torch
import torch.nn as nn
from torch.nn import functional as F
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)

    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)

class Feature_Extractor(nn.Module):
    '''
    Extract landmark and texture features: lms and irs
    '''
    def __init__(self, dims = [64, 128, 256], use_blocks=True):
        super().__init__()
        # backbones of landmark and image feature

        self.use_blocks = use_blocks
        self.ir50 = Backbone(50, 0.0, 'ir') # 50 layers, drop = 0.0, mode = 'ir'
        ir_checkpoints = torch.load('./models/pretrain/ir50.pth', map_location = lambda storage, loc:storage)
        self.ir50.load_state_dict(ir_checkpoints)

        self.face_landback = MobileFaceNet([112, 112], 136) # inputsize = (112, 112), embeddingsize = 136
        landback_checkpoints = torch.load('./models/pretrain/mobilefacenet_model_best.pth.tar',
                                          map_location = lambda storage, loc:storage)
        
        self.face_landback.load_state_dict(landback_checkpoints['state_dict'])
        # do not upgrade parameters in landback
        for param in self.face_landback.parameters():
            param.requires_grad = False
        # conv layers
        self.convs = nn.ModuleList()
        for i in range(len(dims)):
            self.convs.append(nn.Conv2d(in_channels=dims[i], out_channels=dims[i], kernel_size=3, padding=1, stride=2))
        self.last_face_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        

    def forward(self, x):
        '''
        Args:
            x (batch_size, 224, 224): input images
        Returns:
            x_faces (list) : list of len(dim)-scale landmarks
            x_irs (list) : list of len(dim)-scale texture fatures
        '''
        x_face = F.interpolate(x, size = 112) # downsample from 224 to 112
        x_faces = list(self.face_landback(x_face)) # tuple to list
        x_faces[-1] = self.last_face_conv(x_faces[-1])
        for face in x_faces:
            face = _to_channel_last(face)
        x_irs = list(self.ir50(x))
        if not self.use_blocks:
            for i in range(len(x_irs)):
                x_irs[i] = self.convs[i](x_irs[i])
        return x_faces, x_irs
    