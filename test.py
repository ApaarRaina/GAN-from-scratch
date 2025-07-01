import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from generator import Generator
from discriminator import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(20).to(device)
discriminator = Discriminator().to(device)

generator.load_state_dict(torch.load("generator.pth", map_location=device))
discriminator.load_state_dict(torch.load("discriminator.pth", map_location=device))

def create_latent_vector(latent_dim=20):
    return np.random.normal(0, 1, latent_dim).astype(np.float32)


latent_vector = torch.tensor(create_latent_vector(100)).unsqueeze(0).to(device)


generator.eval()
with torch.no_grad():
    generated_image = generator(latent_vector)

# Convert to numpy
output_img = generated_image.squeeze().cpu().numpy()

# Plot
plt.figure(figsize=(6, 6))
plt.imshow(output_img, cmap='gray')
plt.axis('off')
plt.title("Generated Image")
plt.show()

