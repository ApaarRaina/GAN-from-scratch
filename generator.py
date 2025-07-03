import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self,in_features):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(in_features,64*10*10),
            nn.ReLU(),
            nn.Unflatten(1,(64,10,10)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1),
            nn.Tanh()
        )

    def forward(self,noise):

        generated_img=self.net(noise)
        generated_img=F.interpolate(generated_img,(28,28))

        return generated_img