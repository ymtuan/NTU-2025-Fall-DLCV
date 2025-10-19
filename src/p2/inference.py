import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
import argparse
import sys
from glob import glob

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

    # Get all noise files
    noise_files = sorted(glob(os.path.join(args.noise_path, '*.pt')))
    
    for noise_file in noise_files:
        # Get output filename
        base_name = os.path.splitext(os.path.basename(noise_file))[0]
        output_file = os.path.join(args.output_path, f'{base_name}.png')
        
        # Load and process noise
        noise = torch.load(noise_file)
        noise = noise.to(device)

        # Generate images using DDIM
        with torch.no_grad():
            generated_images = sampler.ddim_sample(noise, eta=0.0)
            # Scale from [-1,1] to [0,1] range
            generated_images = (generated_images + 1) / 2.0
            # Clamp to ensure values are in [0,1]
            # generated_images = torch.clamp(generated_images, 0, 1)
            save_image(generated_images, output_file, normalize=True)
        print(f"Generated {output_file}")

if __name__ == '__main__':
    main()