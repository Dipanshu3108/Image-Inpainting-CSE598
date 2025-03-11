import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
from collections import defaultdict

def preprocess_image(image_path: str, size=(64, 64)) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.Resampling.LANCZOS)
    return np.array(img).astype(float) / 255.0

def evaluate_images(original_dir: str, inpainted_dir: str):
    mask_metrics = {
        'circle': defaultdict(list),
        'rectangle': defaultdict(list),
        'square': defaultdict(list),
        'triangle': defaultdict(list)
    }
    
    files = [f for f in os.listdir(original_dir) if f.endswith('.jpg')]
    
    for filename in tqdm(files, desc="Processing images"):
        image_number = filename.split('.')[0]
        original_path = os.path.join(original_dir, filename)
        
        # Process each mask type
        for mask_type in mask_metrics.keys():
            inpainted_path = os.path.join(inpainted_dir, f"{image_number}_{mask_type}_random_inpainted.jpg")
            
            if not os.path.exists(inpainted_path):
                continue
                
            try:
                img1 = preprocess_image(original_path)
                img2 = preprocess_image(inpainted_path)
                
                mask_metrics[mask_type]['ssim'].append(ssim(img1, img2, channel_axis=2, data_range=1.0))
                mask_metrics[mask_type]['psnr'].append(psnr(img1, img2, data_range=1.0))
                mask_metrics[mask_type]['mse'].append(np.mean((img1 - img2) ** 2))
                
            except Exception:
                continue
    
    # Calculate summary statistics
    summary = {}
    for mask_type, metrics in mask_metrics.items():
        if metrics['ssim']:  # Only include if we have data
            summary[mask_type] = {
                'SSIM': f"{np.mean(metrics['ssim']):.3f}±{np.std(metrics['ssim']):.3f}",
                'MSE': f"{np.mean(metrics['mse']):.3f}±{np.std(metrics['mse']):.3f}",
                'PSNR': f"{np.mean(metrics['psnr']):.3f}±{np.std(metrics['psnr']):.3f}"
            }
    
    # Calculate overall metrics
    all_ssim = [v for metrics in mask_metrics.values() for v in metrics['ssim']]
    all_mse = [v for metrics in mask_metrics.values() for v in metrics['mse']]
    all_psnr = [v for metrics in mask_metrics.values() for v in metrics['psnr']]
    
    summary['Overall'] = {
        'SSIM': f"{np.mean(all_ssim):.3f}±{np.std(all_ssim):.3f}",
        'MSE': f"{np.mean(all_mse):.3f}±{np.std(all_mse):.3f}",
        'PSNR': f"{np.mean(all_psnr):.3f}±{np.std(all_psnr):.3f}"
    }
    
    return summary

if __name__ == "__main__":
    metrics = evaluate_images("img_align_celeba", "noise_inpainted_results")
    
    print("\nResults by Mask Type:")
    for mask_type, results in metrics.items():
        print(f"\n{mask_type}:")
        for metric, value in results.items():
            print(f"{metric}: {value}")