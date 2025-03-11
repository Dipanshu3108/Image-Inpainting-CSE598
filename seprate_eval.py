import os
import logging
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Mean Squared Error between two images."""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def load_and_resize_image(image_path: str, target_size=None) -> np.ndarray:
    """Load image and resize if target_size is provided."""
    try:
        img = Image.open(image_path).convert('RGB')
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {str(e)}")
        raise


def evaluate_image_set(original_path: str, noisy_path: str, inpainted_path: str) -> Dict[str, Dict[str, float]]:
    """Calculate metrics for multiple comparison pairs."""
    try:
        
        original = Image.open(original_path).convert('RGB')
        target_size = original.size  
        
        # resize all images to match original
        original = np.array(original)
        noisy = load_and_resize_image(noisy_path, target_size)
        inpainted = load_and_resize_image(inpainted_path, target_size)
        
        logging.info(f"Resized images to {target_size}")
        
        # Calculate metrics for all comparisons
        metrics = {
            'PSNR': {
                'original_vs_noisy': psnr(original, noisy),        # How much noise affected the image? - address this in presentation slide
                'original_vs_inpainted': psnr(original, inpainted),# How well inpainting restored to original?
                'noisy_vs_inpainted': psnr(noisy, inpainted)      # How much inpainting changed the noisy image?
            },
            'SSIM': {
                'original_vs_noisy': ssim(original, noisy, channel_axis=2),
                'original_vs_inpainted': ssim(original, inpainted, channel_axis=2),
                'noisy_vs_inpainted': ssim(noisy, inpainted, channel_axis=2)
            },
            'MSE': {
                'original_vs_noisy': calculate_mse(original, noisy),
                'original_vs_inpainted': calculate_mse(original, inpainted),
                'noisy_vs_inpainted': calculate_mse(noisy, inpainted)
            }
        }
        
        return metrics
    except Exception as e:
        logging.error(f"Error in evaluate_image_set: {str(e)}")
        raise

def create_visualizations(df: pd.DataFrame, output_dir: str) -> None:
    """Create and save visualization plots."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style parameters
        plt.rcParams['figure.figsize'] = (20, 6)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        sns.set_theme(style="whitegrid")
        
        metrics = ['PSNR', 'SSIM', 'MSE']
        fig, axes = plt.subplots(1, 3)
        
        for idx, metric in enumerate(metrics):
            data = pd.melt(
                df,
                id_vars=['noise_type'],
                value_vars=[f'{metric}_original_vs_noisy', f'{metric}_original_vs_inpainted', f'{metric}_noisy_vs_inpainted'],
                var_name='Type',
                value_name=metric
            )
            
            sns.barplot(
                data=data,
                x='noise_type',
                y=metric,
                hue='Type',
                ax=axes[idx]
            )
            
            axes[idx].set_title(f'{metric} Comparison by Noise Type')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for container in axes[idx].containers:
                axes[idx].bar_label(container, fmt='%.2f', padding=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.error(f"Error in create_visualizations: {str(e)}")
        raise


def save_mask_specific_results(df: pd.DataFrame, output_dir: str) -> None:
    """Save results in separate CSVs for each mask type and overall results."""
    try:
        # 1. Overall detailed results
        df.to_csv(os.path.join(output_dir, 'overall_results.csv'), index=False)
        
        # 2. Separate results for each mask type
        mask_types = ['line', 'circle', 'rectangle', 'triangle']
        
        for mask_type in mask_types:
            # Filter data for this mask type
            mask_df = df[df['mask_type'].str.contains(mask_type, case=False)].copy()
            
            if not mask_df.empty:
                # Create filename
                filename = f'{mask_type}_results.csv'
                
                # Save to CSV
                mask_df.to_csv(os.path.join(output_dir, filename), index=False)
                
                # Also create a summary for this mask type
                summary_df = mask_df.groupby('noise_type').agg({
                    'PSNR_original_vs_noisy': ['mean', 'std'],
                    'PSNR_original_vs_inpainted': ['mean', 'std'],
                    'PSNR_noisy_vs_inpainted': ['mean', 'std'],
                    'SSIM_original_vs_noisy': ['mean', 'std'],
                    'SSIM_original_vs_inpainted': ['mean', 'std'],
                    'SSIM_noisy_vs_inpainted': ['mean', 'std'],
                    'MSE_original_vs_noisy': ['mean', 'std'],
                    'MSE_original_vs_inpainted': ['mean', 'std'],
                    'MSE_noisy_vs_inpainted': ['mean', 'std']
                }).round(4)
                
                summary_df.to_csv(os.path.join(output_dir, f'{mask_type}_summary.csv'))
                
        logging.info(f"Saved results for all mask types in {output_dir}")
        
    except Exception as e:
        logging.error(f"Error saving mask-specific results: {str(e)}")
        raise

def main():
    """Main execution function."""
    try:
        # Set up paths
        output_dir = "evaluation_results_mapped"
        os.makedirs(output_dir, exist_ok=True)
        
        original_dir = "img_align_celeba"
        noisy_dir = "noise_image"
        inpainted_dir = "noise_inpainted_results"
        
        results = []
        
        # Process noisy images
        for noisy_file in sorted(os.listdir(noisy_dir)):
            if not noisy_file.endswith('.jpg'):
                continue
            
            parts = noisy_file.split('_')
            if len(parts) < 2:
                logging.warning(f"Skipping malformed filename: {noisy_file}")
                continue
                
            image_number = parts[0]
            noise_type = parts[1].split('.')[0]
            
            logging.info(f"Processing image {image_number} with noise type {noise_type}")
            
            # Get paths
            original_file = f"{image_number}.jpg"
            original_path = os.path.join(original_dir, original_file)
            noisy_path = os.path.join(noisy_dir, noisy_file)
            
            if not os.path.exists(original_path):
                logging.warning(f"Original image not found: {original_path}")
                continue
            
            # Find corresponding inpainted images
            inpainted_pattern = f"{image_number}_{noise_type}"
            inpainted_files = [
                f for f in os.listdir(inpainted_dir)
                if f.startswith(inpainted_pattern)
                and f.endswith('_inpainted.jpg')
            ]
            
            if not inpainted_files:
                logging.warning(f"No inpainted images found for {noisy_file}")
                continue
            
            for inpainted_file in inpainted_files:
                inpainted_path = os.path.join(inpainted_dir, inpainted_file)
                logging.info(f"Processing inpainted file: {inpainted_file}")
                
                try:
                    metrics = evaluate_image_set(original_path, noisy_path, inpainted_path)
                    
                    # Extract mask type from filename
                    parts = inpainted_file.split('_')
                    mask_type = '_'.join(parts[2:-1])  # All parts between noise type and 'inpainted'
                    
                    results.append({
                        'image_number': image_number,
                        'noise_type': noise_type,
                        'mask_type': mask_type,
                        'original_path': original_path,
                        'noisy_path': noisy_path,
                        'inpainted_path': inpainted_path,
                        'PSNR_original_vs_noisy': metrics['PSNR']['original_vs_noisy'],
                        'PSNR_original_vs_inpainted': metrics['PSNR']['original_vs_inpainted'],
                        'PSNR_noisy_vs_inpainted': metrics['PSNR']['noisy_vs_inpainted'],
                        'SSIM_original_vs_noisy': metrics['SSIM']['original_vs_noisy'],
                        'SSIM_original_vs_inpainted': metrics['SSIM']['original_vs_inpainted'],
                        'SSIM_noisy_vs_inpainted': metrics['SSIM']['noisy_vs_inpainted'],
                        'MSE_original_vs_noisy': metrics['MSE']['original_vs_noisy'],
                        'MSE_original_vs_inpainted': metrics['MSE']['original_vs_inpainted'],
                        'MSE_noisy_vs_inpainted': metrics['MSE']['noisy_vs_inpainted']
                    })
                    logging.info(f"Successfully processed {inpainted_file}")
                    
                except Exception as e:
                    logging.error(f"Error processing {inpainted_file}: {str(e)}")
            
        if not results:
            raise ValueError("No results were generated. Check input directories and file patterns.")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save all results
        save_mask_specific_results(df, output_dir)
        
        # Create visualizations
        create_visualizations(df, output_dir)
        
        logging.info("Evaluation completed successfully!")
        logging.info(f"\nResults saved in: {output_dir}")
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()