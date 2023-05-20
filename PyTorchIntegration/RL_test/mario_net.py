# -*- coding: utf-8 -*-
"""
Created on Fri May 19 00:25:20 2023

@author: admin
"""

from importconfig import (
    torch, 
    nn, 
    ToTensor, 
)


# import torch
# from torch import nn
# from torchvision.transforms import ToTensor

import copy



class MarioNet(nn.Module):
    """
    mini CNN struct
        input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        c, h, w = input_dim
        
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), 
            nn.ReLU(), 
            nn.Flatten(), 
            nn.Linear(3136, 512), 
            nn.ReLU(), 
            nn.Linear(512, output_dim), 
        )
        
        self.target = copy.deepcopy(self.online)
        
        # Q_target parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False
        
    def forward(self, input, model):
        if not torch.is_tensor(input):
            input = ToTensor(input)
        
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        else:
            raise ValueError(f"Expecting model: 'online' or 'target', got: {model}")