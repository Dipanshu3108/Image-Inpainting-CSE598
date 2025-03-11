from PIL import Image
from simple_lama_inpainting import SimpleLama
import os
import numpy as np

def process_image_with_masks(model, image_path, masks_dir, output_dir, image_number):
    """Process a single image with all its corresponding masks"""
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    mask_prefix = f"{image_number}_"
    masks = [f for f in os.listdir(masks_dir) if f.startswith(mask_prefix)]
    
    for mask_file in masks:
        # Load and process mask
        mask_path = os.path.join(masks_dir, mask_file)
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
        mask_np = (mask_np > 127).astype(np.uint8) * 255
        
        # Get mask type from filename (e.g., "circle_random")
        mask_type = mask_file.replace(f"{image_number}_", "").replace(".jpg", "")
        
        # Inpaint
        result = model(image_np, mask_np)
        
        output_path = os.path.join(output_dir, f"{image_number}_{mask_type}_inpainted.jpg")
        result.save(output_path)
        
        print(f"Processed {image_number} with mask {mask_type}")

def main():
    model = SimpleLama()
    
    # Set up paths
    image_dir = "img_align_celeba"
    masks_dir =  "img_align_celeba_masks"
    output_dir = "inpainted_results"
    
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    total_images = len(image_files)
    print(f"Found {total_images} images to process")
    
    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        image_number = image_file.split('.')[0]
        
        try:
            process_image_with_masks(model, image_path, masks_dir, output_dir, image_number)
            if idx % 500 == 0: 
                print(f"Processed {idx}/{total_images} images")
        except Exception as e:
            print(f"Error processing image {image_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()