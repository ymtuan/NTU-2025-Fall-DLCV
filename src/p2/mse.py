import os
import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error  # Use the imported function
import argparse
import pathlib

def calculate_mse_for_folders(folder1, folder2):
    """
    Calculate MSE between corresponding images in two folders with a safety check.
    """
    # Verify folders exist
    if not os.path.exists(folder1):
        raise ValueError(f"Ground truth folder {folder1} does not exist.")
    if not os.path.exists(folder2):
        raise ValueError(f"Generated image folder {folder2} does not exist.")

    # Get sorted lists of PNG files
    folder1_files = sorted([f for f in os.listdir(folder1) if f.endswith('.png')])
    folder2_files = sorted([f for f in os.listdir(folder2) if f.endswith('.png')])

    # Check for expected number of files (10 images: 00.png to 09.png)
    expected_files = [f"{i:02d}.png" for i in range(10)]
    if set(folder1_files) != set(expected_files) or set(folder2_files) != set(expected_files):
        raise ValueError(f"Expected 10 images (00.png to 09.png) in both folders, got {len(folder1_files)} and {len(folder2_files)}.")

    mse_list = []
    
    for file1, file2 in zip(folder1_files, folder2_files):
        # Load images and convert to RGB
        img1_pil = Image.open(os.path.join(folder1, file1)).convert('RGB')
        img2_pil = Image.open(os.path.join(folder2, file2)).convert('RGB')
        
        # Convert to numpy arrays using a consistent data type (float32) for accurate math
        img1 = np.array(img1_pil, dtype=np.float32)
        img2 = np.array(img2_pil, dtype=np.float32)
        
        # --- SAFETY CHECK & NORMALIZATION ---
        # If the generated image's max pixel value is <= 1.0, it's likely in the [0, 1] range.
        # This is a common output format for neural networks.
        if np.max(img2) <= 1.0:
            print(f"Info: Scaling '{file2}' from [0, 1] to [0, 255] range.")
            img2 = img2 * 255.0
        
        # Ensure image shapes are identical before comparison
        if img1.shape != img2.shape:
            raise ValueError(f"Different image sizes for {file1} and {file2}. GT is {img1.shape}, Gen is {img2.shape}")
        
        # Calculate MSE using the scikit-image function
        # This is equivalent to np.mean((img1 - img2) ** 2) but is cleaner
        mse = mean_squared_error(img1, img2)
        mse_list.append((file1, mse))
    
    total_mse = np.sum([mse for _, mse in mse_list])
    return mse_list, total_mse

def evaluate_mse(gt_folder, gen_folder, baseline_mse=20.0):
    """
    Evaluate total MSE and determine if it meets the baseline.
    """
    try:
        mse_list, total_mse = calculate_mse_for_folders(gt_folder, gen_folder)
        print("--- MSE Results ---")
        for file_name, mse_val in mse_list:
            print(f"  {file_name}: {mse_val:.4f}")
        print("-------------------")
        print(f"Total MSE: {total_mse:.4f}")

        if total_mse < baseline_mse:
            print(f"Pass: Total MSE ({total_mse:.4f}) is below baseline ({baseline_mse}).")
        else:
            print(f"Fail: Total MSE ({total_mse:.4f}) exceeds baseline ({baseline_mse}).")
        return total_mse
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MSE between ground truth and generated images.")
    parser.add_argument('--gt_dir', type=pathlib.Path, required=False, default='../../hw2_data/face/GT', help="Path to ground truth image directory.")
    parser.add_argument("--gen_dir", type=pathlib.Path, required=False, default='output', help="Path to generated image directory.")
    args = parser.parse_args()

    evaluate_mse(args.gt_dir, args.gen_dir)