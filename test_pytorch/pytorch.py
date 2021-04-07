# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:42:04 2021

@author: TK6YNZ7
"""

import torch

device = 'cuda'

# Global Hyper-parameters
dataset_size = 7481
batch_size = 4
num_workers = 0
shuffle = True
image_width = 512
image_height = 512
num_epochs = 10
lr = 0.0001
K = 1

import os
import numpy as np
from PIL import Image
import torchvision as tv

tensor = tv.transforms.ToTensor()
image = tv.transforms.ToPILImage()

def ConvBlock(in_filters, out_filters, stride, kernel_size, padding = False):
        if padding:
            pad_denom = 4
        else:
            pad_denom = 2
        return torch.nn.Sequential(
            torch.nn.ReflectionPad2d(kernel_size//pad_denom),
            torch.nn.Conv2d(in_channels=in_filters,out_channels=out_filters,kernel_size=kernel_size,stride=stride),           
            torch.nn.BatchNorm2d(num_features=out_filters), 
            torch.nn.ReLU())
  
def ConvTransBlock(in_filters, out_filters, stride, kernel_size, padding):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_filters,out_channels=out_filters,kernel_size=kernel_size,stride=stride,padding = 1,output_padding=padding),
            torch.nn.BatchNorm2d(num_features=out_filters),
            torch.nn.ReLU())
    
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.convblock = ConvBlock(channels, channels, 1,3)
        self.convoactivation = torch.nn.Sequential(
            torch.torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(channels))
        self.leakyrelu = torch.nn.ReLU()
    def forward(self, inputs):
        residual = inputs
        out = self.convblock(inputs)
        out = self.convoactivation(out)
        out = out + residual
        out = self.leakyrelu(out)
        return out

class Encoder(torch.nn.Module):    
    def __init__(self,ngf,initial_filters,initial_channels):
        super(Encoder, self).__init__()   
        self.down = torch.nn.Sequential(
            ConvBlock(initial_channels, ngf, 1, initial_filters),
            ConvBlock(ngf, ngf*2, 2, 3),
            ConvBlock(ngf*2,ngf*4,2,3),
            ConvBlock(ngf*4,ngf*8,2,3),
            ConvBlock(ngf*8,ngf*16,2,3),
            ConvBlock(ngf*16,ngf*32,2,3),
            ConvBlock(ngf*32,ngf*64,2,3),
            ConvBlock(ngf*64,ngf*128,2,3))    
        self.bottleneck = torch.nn.Sequential(
            ResidualBlock(ngf*128),
            ResidualBlock(ngf*128),
            ResidualBlock(ngf*128),
            ResidualBlock(ngf*128),
            ResidualBlock(ngf*128))       
        self.up = torch.nn.Sequential(
            ConvTransBlock(ngf*128, ngf*64, 2, 3,1),
            ConvTransBlock(ngf*64, ngf*32, 2, 3,1),
            ConvTransBlock(ngf*32, ngf*16, 2, 3,1),
            ConvTransBlock(ngf*16, ngf*8, 2, 3,1),
            ConvTransBlock(ngf*8, ngf*4, 2, 3,1),
            ConvTransBlock(ngf*4,ngf*2,2,3,1),
            ConvTransBlock(ngf*2,ngf,2,3,1),
            torch.nn.ReflectionPad2d(initial_filters//2),
            torch.nn.Conv2d(in_channels=ngf,out_channels=initial_channels,kernel_size=initial_filters),
            torch.nn.BatchNorm2d(num_features=initial_channels),
            torch.nn.Sigmoid())
    def forward(self, inputs):
         return self.up(self.bottleneck(self.down(inputs)))

model=Encoder(8,7,3).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = torch.nn.BCELoss()

def trainG(tgt_imgs):
        model.zero_grad()
        tgt_imgs = tgt_imgs.to(device)
        x_hat = model(tgt_imgs.to(device))
        loss = criterion(x_hat,tgt_imgs)
        loss.backward()
        optimizer.step()
        return loss
    
for epoch in range(num_epochs):
    loss = trainG(torch.zeros((2,3,256,256)))     
    print(epoch,loss)