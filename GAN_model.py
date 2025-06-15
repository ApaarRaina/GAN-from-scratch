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

discriminator=Discriminator()

generator.train()
discriminator.train()


epochs=30
criterion=nn.BCELoss()
optimizer_d=torch.optim.AdamW(discriminator.parameters(),lr=0.001)
optimizer_g=torch.optim.AdamW(generator.parameters(),lr=0.001)

for i in range(epochs):

    for j,(cartoon_image,noise_array) in enumerate(zip(cartoon_images,noise_arrays)):

           generated_imgs=generator(noise_array)
           real_labels=torch.ones(100,1)
           fake_labels=torch.zeros(100,1)

           real_output=torch.max(discriminator(cartoon_image),dim=1)[0].unsqueeze(1)
           fake_output=torch.max(discriminator(generated_imgs.detach()),dim=1)[0].unsqueeze(1)

           d_loss_real=criterion(real_output,real_labels)
           d_loss_fake=criterion(fake_output,fake_labels)

           d_loss=d_loss_fake+d_loss_real

           print(f"The discrminator loss is {d_loss.item()} at epoch {i}")

           optimizer_d.zero_grad()
           d_loss.backward()
           optimizer_d.step()

           if i!=0 and i%6==0:
               g_output=torch.argmax(discriminator(generated_imgs))
               g_loss=criterion(g_output,real_labels)

               print(f"The generator loss is {g_loss.item()} at epoch {i}")

               optimizer_g.zero_grad()
               g_loss.backward()
               optimizer_g.step()











