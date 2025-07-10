import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from generator import Generator
from discriminator import Discriminator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

generator = Generator(100).to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))

latent_vector = noise = torch.randn(1, 100, device=device)


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

plt.savefig("Result")

plt.show()



