import pandas as pd
import os

def transform_csv(input_csv_path, output_folder="evaluation_results_mapped\\updated_res\\noise_vs_inpainted"):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the original CSV file
    df = pd.read_csv(input_csv_path)
    
    # Create new dataframe with required columns
    new_df = pd.DataFrame()
    
    # Extract image number for original images
    new_df['img'] = df['image_number'].apply(lambda x: f"{x:06d}.jpg")
    
    # Create inpainted filenames in the new format
    new_df['inpainted'] = df.apply(
        lambda row: f"{row['image_number']:06d}_{row['mask_type']}_inpainted.jpg", 
        axis=1
    )
    
    # Copy the metrics we want
    new_df['ssim'] = df['SSIM_noisy_vs_inpainted']
    new_df['mse'] = df['MSE_noisy_vs_inpainted']
    new_df['psnr'] = df['PSNR_noisy_vs_inpainted']
    
    # Generate output filename
    input_filename = os.path.basename(input_csv_path)
    output_path = os.path.join(output_folder, f"transformed_{input_filename}")
    
    # Save to new CSV file
    new_df.to_csv(output_path, index=False)
    
    print(f"Transformed CSV saved to {output_path}")
    print("\nFirst few rows of the new format:")
    print(new_df.head())
    
    return new_df

# Example usage
if __name__ == "__main__":
    input_csv = "evaluation_results_mapped\\triangle_results.csv"  # Replace with your input CSV path
    transformed_df = transform_csv(input_csv)