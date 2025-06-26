import torch
import torch.nn as nn
import cv2 as cv
import numpy as np


class Generator(nn.Module):

    def __init__(self,in_features):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(in_features,30),
            nn.ReLU(),
            nn.Linear(30,50),
            nn.ReLU(),
            nn.Linear(50,64),
            nn.ReLU()
        )

        self.conv_layers=nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=0),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=0)
         )

    def forward(self,noise):

        flat=self.net(noise)
        batch_size = noise.size(0)
        initial_img=flat.view(batch_size,1,8,8)
        generated_img=self.conv_layers(initial_img)

        return generated_img