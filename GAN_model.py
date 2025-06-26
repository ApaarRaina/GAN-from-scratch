import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Dataset
from generator import Generator
from discriminator import Discriminator
import cv2 as cv
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)


class Noise(Dataset):
    def __init__(self, dataset_path, transform):
        self.path = dataset_path
        self.image_paths = [os.path.join(self.path, fname)
                            for fname in os.listdir(self.path)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return torch.flatten(img)


transform_mnist = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

transform_noise = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

noise_path = "C:\\programs\\GAN\\venv\\normal_dataset"

noise_dataset = Noise(noise_path, transform_noise)
mnist_dataset = datasets.MNIST(root='./content/mnist', train=False, transform=transform_mnist, download=False)

noise_arrays = DataLoader(dataset=noise_dataset, batch_size=1000, shuffle=True,drop_last=True)
mnist_imgs = DataLoader(dataset=mnist_dataset, batch_size=1000, shuffle=True, drop_last=True)

generator = Generator(20).to(device)
discriminator = Discriminator().to(device)

generator.train()
discriminator.train()

epochs = 5
criterion = nn.BCELoss()
optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=0.0003)
optimizer_g = torch.optim.AdamW(generator.parameters(), lr=0.001)

for i in range(epochs):
    batch_no = 0
    for j, ((mnist_img, _), noise_array) in enumerate(zip(mnist_imgs, noise_arrays)):
        mnist_img = mnist_img.to(device)
        noise_array = noise_array.to(device)
        generated_imgs = generator(noise_array)
        batch_size = mnist_img.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        real_output = torch.max(discriminator(mnist_img), dim=1)[0].unsqueeze(1)
        fake_output = torch.max(discriminator(generated_imgs.detach()), dim=1)[0].unsqueeze(1)

        d_loss_real = criterion(real_output, real_labels)
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss = d_loss_fake + d_loss_real

        print(f"The discriminator loss is {d_loss.item()} at epoch {i}, batch {batch_no}")

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        if batch_no != 0 and batch_no % 2 == 0:

            for iter in range(4):
                generated_imgs = generator(noise_array)
                g_output = torch.max(discriminator(generated_imgs), dim=1)[0].unsqueeze(1)
                g_loss = criterion(g_output, real_labels)

                print(f"The generator loss is {g_loss.item()} at epoch {i}, batch {batch_no}")

                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()

        batch_no += 1
