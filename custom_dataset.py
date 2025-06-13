import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_normal_image(mean=128, std=20, size=(500, 500)):
    # Generate normally distributed pixels
    img_array = np.random.normal(loc=mean, scale=std, size=size)

    # Clip to valid pixel range and convert to uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    return Image.fromarray(img_array, mode='L')  # 'L' for grayscale

# Create directory to save images
os.makedirs("normal_dataset", exist_ok=True)

# Generate dataset
for i in tqdm(range(10000), desc="Generating images"):
    img = create_normal_image()
    img.save(f"normal_dataset/image_{i:05d}.png")
