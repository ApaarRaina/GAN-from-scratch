import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)


def create_latent_vector(latent_dim=20):
    return np.random.normal(0, 1, latent_dim).astype(np.float32)


transform_mnist = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = datasets.MNIST(root='./content/mnist', train=False, transform=transform_mnist, download=True)

mnist_imgs = DataLoader(dataset=mnist_dataset, batch_size=1000, shuffle=True, drop_last=True)

generator = Generator(100).to(device)
discriminator = Discriminator().to(device)

generator.load_state_dict(torch.load("/content/generator.pth", map_location=device))
discriminator.load_state_dict(torch.load("/content/discriminator.pth", map_location=device))

generator.train()
discriminator.train()

epochs = 20
criterion = nn.BCELoss()
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0004)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)

k = 1
for i in range(epochs):
    batch_no = 0
    for j, (mnist_img, _) in enumerate(mnist_imgs):

        mnist_img = mnist_img.to(device)
        noise_array = noise = torch.randn(1000, 100, device=device).to(device)
        generated_imgs = generator(noise_array)
        batch_size = mnist_img.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        real_output = discriminator(mnist_img)
        fake_output = discriminator(generated_imgs.detach())

        d_loss_real = criterion(real_output, real_labels)
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss = d_loss_fake + d_loss_real

        print(f"The discriminator loss is {d_loss.item()} at epoch {i}, batch {batch_no}")

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        for j in range(k):
            generated_imgs = generator(noise_array)
            g_output = discriminator(generated_imgs)
            g_loss = criterion(g_output, real_labels)

            print(f"The generator loss is {g_loss.item()} at epoch {i}, batch {batch_no}")

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        batch_no += 1
