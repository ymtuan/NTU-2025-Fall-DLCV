# eval_p2.py
import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as tf

# --- Imports from InstantSplat environment ---
# We use the specific local module instead of standard 'lpips'
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr

def read_image(path, device):
    """Loads an image, converts to tensor (0-1), sends to device."""
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('RGB')
    # Convert to tensor (0.0 to 1.0 range)
    tensor = tf.to_tensor(img).unsqueeze(0).to(device)
    return tensor

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get File Lists
    if not os.path.exists(args.renders_dir) or not os.path.exists(args.gt_dir):
        print(f"Error: One of the directories does not exist.")
        print(f"Renders: {args.renders_dir}")
        print(f"GT:      {args.gt_dir}")
        return

    render_files = sorted(os.listdir(args.renders_dir))
    
    psnr_list = []
    ssim_list = []
    lpips_list = []

    print(f"\nEvaluating Renders: {args.renders_dir}")
    print(f"Ground Truth Dir:   {args.gt_dir}")

    count = 0
    
    for fname in tqdm(render_files, desc="Calculating Metrics"):
        if not fname.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        render_path = os.path.join(args.renders_dir, fname)
        gt_path = os.path.join(args.gt_dir, fname)

        # 1. Load Images
        pred = read_image(render_path, device)
        gt = read_image(gt_path, device)

        if pred is None or gt is None:
            # Skip if GT doesn't exist
            continue

        # 2. Calculate Metrics using InstantSplat modules
        
        # A. PSNR
        p = psnr(pred, gt).item()
        
        # B. SSIM
        s = ssim(pred, gt).item()
        
        # C. LPIPS (Using lpipsPyTorch from your env)
        # Note: lpipsPyTorch typically expects inputs in [0, 1] range directly
        l = lpips(pred, gt, net_type='vgg').item()

        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l)
        count += 1

    if count == 0:
        print("\n[Error] No matching ground truth images found.")
        print("Check if your GT folder paths are correct.")
        return

    # 3. Final Report
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)

    print("\n" + "="*40)
    print(f" FINAL RESULTS (Computed on {count} images)")
    print("="*40)
    print(f" PSNR  : {avg_psnr:.4f}")
    print(f" SSIM  : {avg_ssim:.4f}")
    print(f" LPIPS : {avg_lpips:.4f} (VGG)")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW4 P2 Evaluation")
    parser.add_argument("--renders_dir", type=str, required=True) 
    parser.add_argument("--gt_dir", type=str, required=True)
    args = parser.parse_args()
    evaluate(args)