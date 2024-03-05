#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBP implementations using Python and PyTorch
lbp_py : lbp using Pyton 
lbp_pt : lbp using PyTorch

for details:
https://journalengineering.fe.up.pt/index.php/upjeng/article/view/2183-6493_007-004_0005/567

"""

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from .ir50 import Backbone

#------------------------------------------------------------------------------
def lbp_py(Im): 
    
    sat=len(Im)
    sut=len(Im[0])
    L=np.zeros((sat,sut))
    I=np.zeros((sat+2,sut+2)) 
    I[1:sat+1,1:sut+1]=Im
    for i in range(1,sat+1):
        for j in range(1,sut+1):
            L[i-1,j-1]=\
            ( I[i-1,j]  >= I[i,j] )*1+\
            ( I[i-1,j+1]>= I[i,j] )*2+\
            ( I[i,j+1]  >= I[i,j] )*4+\
            ( I[i+1,j+1]>= I[i,j] )*8+\
            ( I[i+1,j]  >= I[i,j] )*16+\
            ( I[i+1,j-1]>= I[i,j] )*32+\
            ( I[i,j-1]  >= I[i,j] )*64+\
            ( I[i-1,j-1]>= I[i,j] )*128;  
    
    return L
#------------------------------------------------------------------------------


def lbp_pt(x):
    #pad image for 3x3 mask size
    x = F.pad(input=x, pad = [1, 1, 1, 1], mode='constant')
    b=x.shape
    M=b[1]
    N=b[2]
    
    y=x
    #select elements within 3x3 mask 
    # y00  y01  y02
    # y10  y11  y12
    # y20  y21  y22
    
    y00=y[:,0:M-2, 0:N-2]
    y01=y[:,0:M-2, 1:N-1]
    y02=y[:,0:M-2, 2:N  ]
    #     
    y10=y[:,1:M-1, 0:N-2]
    y11=y[:,1:M-1, 1:N-1]
    y12=y[:,1:M-1, 2:N  ]
    #
    y20=y[:,2:M, 0:N-2]
    y21=y[:,2:M, 1:N-1]
    y22=y[:,2:M, 2:N ]      
    
       
    
    # Comparisons 
    # 1 ---------------------------------
    bit=torch.ge(y01,y11)
    tmp=torch.mul(bit,torch.tensor(1)) 
    
    # 2 ---------------------------------
    bit=torch.ge(y02,y11)
    val=torch.mul(bit,torch.tensor(2))
    val=torch.add(val,tmp)    
    
    # 3 ---------------------------------
    bit=torch.ge(y12,y11)
    tmp=torch.mul(bit,torch.tensor(4))
    val=torch.add(val,tmp)
    
    # 4 --------------------------------- 
    bit=torch.ge(y22,y11)
    tmp=torch.mul(bit,torch.tensor(8))   
    val=torch.add(val,tmp)
    
    # 5 ---------------------------------
    bit=torch.ge(y21,y11)
    tmp=torch.mul(bit,torch.tensor(16))   
    val=torch.add(val,tmp)
    
    # 6 ---------------------------------
    bit=torch.ge(y20,y11)
    tmp=torch.mul(bit,torch.tensor(32))   
    val=torch.add(val,tmp)
    
    # 7 ---------------------------------
    bit=torch.ge(y10,y11)
    tmp=torch.mul(bit,torch.tensor(64))   
    val=torch.add(val,tmp)
    
    # 8 ---------------------------------
    bit=torch.ge(y00,y11)
    tmp=torch.mul(bit,torch.tensor(128))   
    val=torch.add(val,tmp)    
    return val

def lbp_extractor_rgb(images):
    B, C, H, W = images.shape
    lbp = torch.zeros((B, H, W, C))
    images = images.permute(0, 2, 3, 1)
    lbp[:,:,:,0]=lbp_pt(images[:,:,:,0])
    lbp[:,:,:,1]=lbp_pt(images[:,:,:,1])
    lbp[:,:,:,2]=lbp_pt(images[:,:,:,2])
    return lbp.permute(0, 3, 1, 2).cuda()

class LBP_Extractor(nn.Module):
    def __init__(self, dims = [64, 128, 256]):
        super().__init__()
        self.dims = dims
        self.ir50 = Backbone(50, 0.0, 'ir',) # 50 layers, drop = 0.0, mode = 'ir'
        ir_checkpoints = torch.load('./models/pretrain/ir50.pth', map_location = lambda storage, loc:storage)
        self.ir50.load_state_dict(ir_checkpoints)
        self.convs = nn.ModuleList(nn.Conv2d(in_channels=dims[i], out_channels=dims[i], kernel_size=3, padding=1, stride=2) for i in range(len(dims)))
        
        
    def forward(self, x):
        #print(x.device)
        lbps = list(self.ir50(lbp_extractor_rgb(x)))
        #print(lbps[0].device)
        lbps = [self.convs[i](lbps[i]) for i in range(len(self.dims))]
        return lbps