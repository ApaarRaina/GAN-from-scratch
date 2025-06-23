import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_layers=nn.Sequential(
             nn.Conv2d(in_channels=1,out_channels=3,kernel_size=4,stride=2),
             nn.Conv2d(in_channels=3,out_channels=5,kernel_size=4,stride=2),
             nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.net=nn.Sequential(
            nn.LazyLinear(200),
            nn.ReLU(),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.Sigmoid()
        )

    def forward(self,img):

        features=self.conv_layers(img)
        features = torch.flatten(features, start_dim=1)
        label=self.net(features)

        return label