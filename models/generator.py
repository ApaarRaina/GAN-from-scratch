import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self,in_features):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(in_features,128*4*4),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1,(128,4,4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self,noise):

        generated_img=self.net(noise)
        generated_img=F.interpolate(generated_img,(28,28))

        return generated_img