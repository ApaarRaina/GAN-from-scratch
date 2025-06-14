import torch
import torch.nn as nn
import cv2 as cv
import numpy as np


class Generator(nn.Module):

    def __init__(self,in_features):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(in_features,300),
            nn.Relu(),
            nn.Linear(300,500),
            nn.Relu(),
            nn.Linear(500,900),
            nn.Relu()
        )

        self.conv_layers=nn.Sequential(
            nn.ConvTranspose2d(in_channels=1,out_channels=6,kernel_size=4,stride=2),
            nn.ConvTranspose2d(in_channels=6,out_channels=8,kernel_size=4,stride=2),
            nn.ConvTranspose2d(in_channels=8,out_channels=12,kernel_size=4,stride=2),
            nn.ConvTranspose2d(in_channels=12,out_channels=18,kernel_size=5,stride=2)
        )

    def forward(self,noise):

        flat=self.net(noise)
        initial_img=flat.view(flat,(30,30))
        generated_img=self.conv_layers(initial_img)
        generated_img=generated_img

        return generated_img


