import os
import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error
import argparse
import pathlib

def calculate_mse_for_folders(folder1, folder2):
    """
    Calculate MSE between corresponding images in two folders, keeping values in [0, 255].
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
        img1_path = os.path.join(folder1, file1)
        img2_path = os.path.join(folder2, file2)

        # Load images and convert to RGB, keeping values in [0, 255]
        img1 = np.array(Image.open(img1_path).convert('RGB'))
        img2 = np.array(Image.open(img2_path).convert('RGB'))
        
        if img1.shape != img2.shape:
            raise ValueError(f"Different image sizes for {file1} and {file2}: {img1.shape} vs {img2.shape}")
        
        # Compute MSE directly on [0, 255] scale
        mse = mean_squared_error(img1, img2)
        mse_list.append((file1, mse))
    
    average_mse = np.mean([mse for _, mse in mse_list])
    return mse_list, average_mse

def evaluate_mse(gt_folder, gen_folder, baseline_mse=20.0):
    """
    Evaluate MSE and determine if it meets the baseline.
    """
    try:
        mse_list, average_mse = calculate_mse_for_folders(gt_folder, gen_folder)
        print(f"MSE per image: {mse_list}")
        print(f"Average MSE: {average_mse:.4f}")

        if average_mse < baseline_mse:
            print(f"Pass: Average MSE ({average_mse:.4f}) is below baseline ({baseline_mse}).")
        else:
            print(f"Fail: Average MSE ({average_mse:.4f}) exceeds baseline ({baseline_mse}).")
        return average_mse
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