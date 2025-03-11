from PIL import Image
from simple_lama_inpainting import SimpleLama
import os
import numpy as np

def process_noisy_image_with_masks(model, noisy_image_path, masks_dir, output_dir, image_number, noise_type):
    """Process a single noisy image with all its corresponding masks"""
    # Load noisy image
    image = Image.open(noisy_image_path).convert('RGB')
    image_np = np.array(image)
    
    mask_prefix = f"{image_number}_"
    masks = [f for f in os.listdir(masks_dir) if f.startswith(mask_prefix)]
    
    for mask_file in masks:
        # Load and process mask
        mask_path = os.path.join(masks_dir, mask_file)
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
        mask_np = (mask_np > 127).astype(np.uint8) * 255
        
    
        mask_type = mask_file.replace(f"{image_number}_", "").replace(".jpg", "")
        
        # Inpaint
        result = model(image_np, mask_np)
        
        output_path = os.path.join(output_dir, f"{image_number}_{noise_type}_{mask_type}_inpainted.jpg")
        result.save(output_path)
        
        print(f"Processed noisy image {image_number} ({noise_type}) with mask {mask_type}")

def main():

    model = SimpleLama()
    
    # Set up paths
    noise_dir = "noise_image"
    masks_dir = "img_align_celeba_masks"
    output_dir = "noise_inpainted_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    noisy_images = sorted([f for f in os.listdir(noise_dir) if f.endswith('.jpg')])
    
    total_images = len(noisy_images)
    print(f"Found {total_images} noisy images to process")
    
    # Process each noisy image
    for idx, noisy_image_file in enumerate(noisy_images, 1):
        # Extract image number and noise type from filename
        # format: "000001_gaussian.jpg"
        parts = noisy_image_file.split('_')
        image_number = parts[0]
        noise_type = parts[1].replace('.jpg', '')
        
        noisy_image_path = os.path.join(noise_dir, noisy_image_file)
        
        try:
            process_noisy_image_with_masks(
                model, 
                noisy_image_path, 
                masks_dir, 
                output_dir, 
                image_number,
                noise_type
            )
            
            if idx % 500 == 0: 
                print(f"Processed {idx}/{total_images} noisy images")
                
        except Exception as e:
            print(f"Error processing noisy image {noisy_image_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()