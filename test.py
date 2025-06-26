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

transform_noise_image = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

def create_normal_image(mean=128, std=20, size=20):
    img_array = np.random.normal(loc=mean, scale=std, size=size)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array, mode='L')

# Create synthetic noise input and flatten it properly
noise_img = create_normal_image()
input_tensor = transform_noise_image(noise_img).flatten().unsqueeze(0).to(device)  # Flatten to 1D
print(f"Input tensor shape: {input_tensor.shape}")

# Generate image using generator
generator.eval()
with torch.no_grad():
    generated_image = generator(input_tensor)

# Convert to numpy
output_img = generated_image.squeeze().cpu().numpy()

# Plot
plt.figure(figsize=(6, 6))
plt.imshow(output_img, cmap='gray')
plt.axis('off')
plt.title("Generated Image")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(noise_img, cmap='gray')
plt.axis('off')
plt.title("Noise Image")
plt.show()

