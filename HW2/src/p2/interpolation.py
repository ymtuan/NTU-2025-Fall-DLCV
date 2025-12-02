import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import argparse
import sys
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math

from model import UNet
from ddim_sampling import DDIMSampler

def slerp(x0, x1, alpha):
    """Spherical Linear Interpolation (SLERP)"""
    theta = torch.acos(torch.clamp(torch.sum(x0 * x1) / (torch.norm(x0) * torch.norm(x1)), -1.0, 1.0))
    sin_theta = torch.sin(theta)
    
    w0 = torch.sin((1 - alpha) * theta) / sin_theta
    w1 = torch.sin(alpha * theta) / sin_theta
    
    return w0 * x0 + w1 * x1

def lerp(x0, x1, alpha):
    """Linear Interpolation (LERP)"""
    return (1 - alpha) * x0 + alpha * x1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_path', type=str, default='../../hw2_data/face/noise',
                      help='Directory containing noise files')
    parser.add_argument('--output_path', type=str, default='./output',
                      help='Directory to save the generated images')
    parser.add_argument('--ckpt_path', type=str, default='../../hw2_data/face/UNet.pt',
                      help='Path to the model checkpoint')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = UNet()
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # Initialize DDIM sampler
    sampler = DDIMSampler(model)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load noise 00 and 01
    noise_00_file = os.path.join(args.noise_path, '00.pt')
    noise_01_file = os.path.join(args.noise_path, '01.pt')
    
    noise_00 = torch.load(noise_00_file).to(device)
    noise_01 = torch.load(noise_01_file).to(device)
    
    alpha_values = [i * 0.1 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
    
    # Generate with SLERP
    print("Generating SLERP interpolation...")
    slerp_images = []
    for alpha in alpha_values:
        interpolated_noise = slerp(noise_00, noise_01, alpha)
        with torch.no_grad():
            generated_image = sampler.ddim_sample(interpolated_noise, eta=0.0)
            generated_image = (generated_image + 1) / 2.0
            generated_image = torch.clamp(generated_image, 0, 1)
            slerp_images.append(generated_image)
    print(f"Generated {len(slerp_images)} SLERP images")
    
    # Generate with LERP
    print("Generating LERP interpolation...")
    lerp_images = []
    for alpha in alpha_values:
        interpolated_noise = lerp(noise_00, noise_01, alpha)
        with torch.no_grad():
            generated_image = sampler.ddim_sample(interpolated_noise, eta=0.0)
            generated_image = (generated_image + 1) / 2.0
            generated_image = torch.clamp(generated_image, 0, 1)
            lerp_images.append(generated_image)
    print(f"Generated {len(lerp_images)} LERP images")
    
    # Create grids for SLERP and LERP
    slerp_grid = make_grid(torch.cat(slerp_images, dim=0), nrow=11, normalize=False)
    lerp_grid = make_grid(torch.cat(lerp_images, dim=0), nrow=11, normalize=False)
    
    # Convert to PIL Images
    slerp_pil = Image.fromarray((slerp_grid.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    lerp_pil = Image.fromarray((lerp_grid.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    
    # Add labels to SLERP
    margin_top = 0
    margin_left = 0
    final_width = slerp_pil.size[0] + margin_left
    final_height = slerp_pil.size[1] + margin_top
    
    final_slerp = Image.new('RGB', (final_width, final_height), color='white')
    final_slerp.paste(slerp_pil, (margin_left, margin_top))
    
    # Add labels to LERP
    final_lerp = Image.new('RGB', (final_width, final_height), color='white')
    final_lerp.paste(lerp_pil, (margin_left, margin_top))

    # Save both grids
    slerp_output = os.path.join(args.output_path, 'slerp.png')
    lerp_output = os.path.join(args.output_path, 'lerp.png')
    
    final_slerp.save(slerp_output)
    final_lerp.save(lerp_output)
    
    print(f"Saved SLERP grid to {slerp_output}")
    print(f"Saved LERP grid to {lerp_output}")

if __name__ == '__main__':
    main()