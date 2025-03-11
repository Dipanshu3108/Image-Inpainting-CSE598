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
    noise_types = ['uniform', 'gaussian', 'exponential', 'gamma']
    mask_types = ['circle', 'rectangle', 'square', 'triangle']
    
    metrics = {noise: {mask: defaultdict(list) for mask in mask_types} 
              for noise in noise_types}
    
    files = [f for f in os.listdir(original_dir) if f.endswith('.jpg')]
    
    for filename in tqdm(files, desc="Processing images"):
        image_number = filename.split('.')[0]
        original_path = os.path.join(original_dir, filename)
        
        for noise_type in noise_types:
            for mask_type in mask_types:
                inpainted_path = os.path.join(inpainted_dir, 
                    f"{image_number}_{noise_type}_{mask_type}_random_inpainted.jpg")
                
                if not os.path.exists(inpainted_path):
                    continue
                    
                try:
                    img1 = preprocess_image(original_path)
                    img2 = preprocess_image(inpainted_path)
                    
                    metrics[noise_type][mask_type]['ssim'].append(
                        ssim(img1, img2, channel_axis=2, data_range=1.0))
                    metrics[noise_type][mask_type]['psnr'].append(
                        psnr(img1, img2, data_range=1.0))
                    metrics[noise_type][mask_type]['mse'].append(
                        np.mean((img1 - img2) ** 2))
                    
                except Exception:
                    continue
    
    summary = {}
    for noise_type in noise_types:
        summary[noise_type] = {}
        for mask_type in mask_types:
            m = metrics[noise_type][mask_type]
            if m['ssim']:
                summary[noise_type][mask_type] = {
                    'SSIM': f"{np.mean(m['ssim']):.3f}±{np.std(m['ssim']):.3f}",
                    'MSE': f"{np.mean(m['mse']):.3f}±{np.std(m['mse']):.3f}",
                    'PSNR': f"{np.mean(m['psnr']):.3f}±{np.std(m['psnr']):.3f}"
                }
    
    return summary

if __name__ == "__main__":
    metrics = evaluate_images("img_align_celeba", "noise_inpainted_results")
    
    for noise_type, mask_results in metrics.items():
        print(f"\n{noise_type.upper()} Noise:")
        for mask_type, results in mask_results.items():
            if results:
                print(f"\n{mask_type}:")
                for metric, value in results.items():
                    print(f"{metric}: {value}")