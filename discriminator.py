import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_layers=nn.Sequential(
             nn.Conv2d(in_channels=1,out_channels=3,kernel_size=4,stride=2),
             nn.MaxPool2d(kernel_size=2,stride=2),
             nn.Conv2d(in_channels=3,out_channels=5,kernel_size=4,stride=2),
             nn.MaxPool2d(kernel_size=2, stride=2),
             nn.Conv2d(in_channels=5,out_channels=8,kernel_size=3,stride=2),
             nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dummy pass to get flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 500, 500)
            dummy_out = self.conv_layers(dummy)
            self.flattened_size = dummy_out.view(1, -1).shape[1]

        self.net=nn.Sequential(
            nn.Linear(self.flattened_size,500),
            nn.ReLU(),
            nn.Linear(500,300),
            nn.ReLU(),
            nn.Linear(300,100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Sigmoid()
        )

    def forward(self,img):

        features=self.conv_layers(img)
        features=features.view(img.shape[0],-1)
        label=self.net(features)

        return label







