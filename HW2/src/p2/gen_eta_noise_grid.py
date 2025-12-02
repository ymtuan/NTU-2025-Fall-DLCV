import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import argparse
import sys
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from model import UNet
from ddim_sampling import DDIMSampler

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

    # Generate with specific noise files and eta values
    noise_indices = ['00', '01', '02', '03']
    eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Store all generated images: [eta_idx][noise_idx]
    all_images = []
    
    for eta in eta_values:
        eta_images = []
        for noise_idx in noise_indices:
            noise_file = os.path.join(args.noise_path, f'{noise_idx}.pt')
            
            # Load and process noise
            noise = torch.load(noise_file)
            noise = noise.to(device)

            # Generate images using DDIM
            with torch.no_grad():
                generated_images = sampler.ddim_sample(noise, eta=eta)
                # Scale from [-1,1] to [0,1] range
                generated_images = (generated_images + 1) / 2.0
                # Clamp to ensure values are in [0,1]
                generated_images = torch.clamp(generated_images, 0, 1)
                eta_images.append(generated_images)
        
        all_images.append(eta_images)
        print(f"Generated images for eta={eta}")
    
    # Create grid: 5 rows (eta values) x 4 columns (noise files)
    grid_images = []
    for row in all_images:
        for img in row:
            grid_images.append(img)
    
    grid_images = torch.cat(grid_images, dim=0)
    grid = make_grid(grid_images, nrow=4, normalize=False)
    
    # Convert to PIL Image for adding labels
    grid_pil = Image.fromarray((grid.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    
    # Get image dimensions
    img_width, img_height = grid_pil.size
    
    # Add margins for labels
    margin_left = 80
    margin_top = 60
    final_width = img_width + margin_left
    final_height = img_height + margin_top
    
    # Create new image with white background
    final_image = Image.new('RGB', (final_width, final_height), color='white')
    final_image.paste(grid_pil, (margin_left, margin_top))
    
    # Draw labels
    draw = ImageDraw.Draw(final_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Column labels (noise indices)
    img_per_row = grid_pil.size[0] // 4
    for i, noise_idx in enumerate(noise_indices):
        x = margin_left + (i + 0.5) * img_per_row
        y = margin_top // 2
        draw.text((x - 10, y - 8), f'{noise_idx}.pt', fill='black', font=font)
    
    # Row labels (eta values)
    img_per_col = grid_pil.size[1] // 5
    for i, eta in enumerate(eta_values):
        x = margin_left // 2
        y = margin_top + (i + 0.5) * img_per_col
        draw.text((x - 30, y - 8), f'η={eta}', fill='black', font=font)
    
    # Save labeled grid
    output_file = os.path.join(args.output_path, 'eta_grid.png')
    final_image.save(output_file)
    print(f"Saved labeled grid to {output_file}")

if __name__ == '__main__':
    main()