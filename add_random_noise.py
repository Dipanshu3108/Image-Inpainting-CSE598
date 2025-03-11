import os
import numpy as np
from PIL import Image
import random

def add_gaussian_noise(image, mean=0, std=25):
    noisy = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, noisy.shape)
    noisy = noisy + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_uniform_noise(image, low=-50, high=50):
    noisy = np.array(image).astype(np.float32)
    noise = np.random.uniform(low, high, noisy.shape)
    noisy = noisy + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_exponential_noise(image, scale=25):
    noisy = np.array(image).astype(np.float32)
    noise = np.random.exponential(scale, noisy.shape)
    noisy = noisy + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_gamma_noise(image, shape=2, scale=25):
    noisy = np.array(image).astype(np.float32)
    noise = np.random.gamma(shape, scale, noisy.shape)
    noisy = noisy + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

image_dir = "img_align_celeba"
output_dir = "noise_image"

os.makedirs(output_dir, exist_ok=True)
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

noise_functions = {
    'gaussian': add_gaussian_noise,
    'uniform': add_uniform_noise,
    'exponential': add_exponential_noise,
    'gamma': add_gamma_noise
}

total_images = len(image_files)
images_per_noise = total_images // len(noise_functions)

random.shuffle(image_files)

print(f"Total images: {total_images}")
print(f"Images per noise type: {images_per_noise}")

for idx, image_file in enumerate(image_files):
    noise_type = list(noise_functions.keys())[idx // images_per_noise]
    if idx // images_per_noise >= len(noise_functions):
        continue
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert('RGB')
    
    image_number = image_file.split('.')[0]
    noise_func = noise_functions[noise_type]
    noisy_image = noise_func(image)
    
    noisy_image = Image.fromarray(noisy_image)

    output_path = os.path.join(output_dir, f"{image_number}_{noise_type}.jpg")
    noisy_image.save(output_path)
    
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1} images")

print("Processing complete!")