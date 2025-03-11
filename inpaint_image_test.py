# single image testing to see if the inpainting works
from PIL import Image
from simple_lama_inpainting import SimpleLama
import os
import numpy as np
# Initialize the model
model = SimpleLama()

# Paths to your directories
image_dir = "img_align_celeba"
mask_dir = "img_align_celeba_masks"
output_dir = "inpainted_results" 


os.makedirs(output_dir, exist_ok=True)

# Test 
image_number = "000002"
mask_shape = "triangle"
mask_type = "random"

# Construct file paths
image_path = os.path.join(image_dir, f"{image_number}.jpg")
mask_path = os.path.join(mask_dir, f"{image_number}_{mask_shape}_{mask_type}.jpg")
output_path = os.path.join(output_dir, f"{image_number}_inpainted.jpg")

# Load images
image = Image.open(image_path).convert('RGB')
mask = Image.open(mask_path).convert('L')

image_np = np.array(image)
mask_np = np.array(mask)

mask_np = (mask_np > 127).astype(np.uint8) * 255

# Apply inpainting
result = model(image_np, mask_np)

# Save result
result.save(output_path)
print(f"Inpainting complete! Result saved as: {output_path}")