import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from discriminator import Discriminator
from generator import Generator
import cv2 as cv
from PIL import Image

class Cartoon(Dataset):

    def __init__(self,transform,dataset_path):
       self.path=dataset_path
       self.transform=transform
       self.image_paths= [os.path.join(self.path, fname)
                            for fname in os.listdir(self.path)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        img_path=self.image_paths[idx]
        img=cv.imread(img_path)
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

        img=Image.fromarray(img)

        if self.transform:
            img=self.transform(img)

        return img



class Noise(Dataset):

    def __init__(self,dataset_path,transform):
       self.path=dataset_path
       self.image_paths= [os.path.join(self.path, fname)
                            for fname in os.listdir(self.path)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
       self.transform=transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        img_path=self.image_paths[idx]
        img=cv.imread(img_path)
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

        img=Image.fromarray(img)

        if self.transform:
            img=self.transform(img)

        return torch.flatten(img)


class GAN_loss(nn.Module):

    def __init__(self):
        super(GAN_loss,self).__init__()
        pass

transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

dataset_path="C:\\programs\\GAN\\venv\\cartoonset10k"
noise_path="C:\\programs\\GAN\\venv\\normal_dataset"

cartoon_dataset=Cartoon(transform,dataset_path)
noise_dataset=Noise(noise_path,transform)

cartoon_images=DataLoader(dataset=cartoon_dataset,batch_size=100,shuffle=True)
noise_arrays=DataLoader(dataset=noise_dataset,batch_size=100,shuffle=True)

generator=Generator(100)

discrminator=Discriminator()

epochs=30

generator.train()
discrminator.train()



