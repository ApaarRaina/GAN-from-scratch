import torch
import torch.nn as nn
import cv2 as cv
import numpy as np


class Generator(nn.Module):

    def __init__(self,in_features):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(in_features,150),
            nn.ReLU(),
            nn.Linear(150,200),
            nn.ReLU(),
            nn.Linear(200,225),
            nn.ReLU()
        )

        self.conv_layers=nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 1, kernel_size=9, stride=1, padding=0)
        )

    def forward(self,noise):

        flat=self.net(noise)
        batch_size = noise.size(0)
        initial_img=flat.view(batch_size,1,15,15)
        generated_img=self.conv_layers(initial_img)

        return generated_img