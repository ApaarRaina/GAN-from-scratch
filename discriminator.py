import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.net=nn.Sequential(

             nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=2),
             nn.LeakyReLU(0.2),
             nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1),
             nn.BatchNorm2d(32),
             nn.LeakyReLU(0.2),
             nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1),
             nn.BatchNorm2d(64),
             nn.LeakyReLU(0.2),
             nn.Flatten(),
             nn.Linear(64*9*9,1),
             nn.Sigmoid()
        )

    def forward(self,img):

        label=self.net(img)

        return label