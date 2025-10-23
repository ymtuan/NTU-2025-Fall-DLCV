import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as trns
from torchvision.utils import save_image, make_grid
import os
from PIL import Image
import csv
import argparse
import pathlib
from datetime import datetime
from model import DDPM, ContextUnet
from torch.cuda.amp import autocast

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImageDataset(Dataset):
    def __init__(self, file_path, csv_path, transform=None):
        self.csv_path = csv_path
        self.path = file_path
        self.transform = transform or trns.Compose([
            trns.Resize([28, 28]),
            trns.ToTensor(),
            trns.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.imgname_csv = []
        self.labels_csv = []
        self.files = []
        self.labels = []
        with open(self.csv_path, "r", newline="") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)
            for row in reader:
                img_name, label = row
                self.imgname_csv.append(img_name)
                self.labels_csv.append(torch.tensor(int(label)))

        for x in os.listdir(self.path):
            if x.endswith(".png") and x in self.imgname_csv:
                self.files.append(os.path.join(self.path, x))
                self.labels.append(self.labels_csv[self.imgname_csv.index(x)])

    def __getitem__(self, idx):
        data = Image.open(self.files[idx])
        data = self.transform(data)
        return data, self.labels[idx]

    def __len__(self):
        return len(self.files)

def generate_100_grid(save_dir, model_path):
    """Generate 100 images (10 per digit) and save as a single grid"""
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = 10

    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=256, n_classes=10),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.0,
    )
    ddpm.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ddpm.to(device)

    os.makedirs(save_dir, exist_ok=True)

    ddpm.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        # Generate 500 samples (same as inference.py)
        n_sample = 50 * n_classes
        with torch.autocast(device_type=device):
            x_gen, _ = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=2.0)

        # Organize images by digit: 10 rows x 10 columns
        grid_images = []
        first_zero = None
        first_one = None
        
        for digit in range(10):
            digit_images = []
            # Get first 10 images for this digit (same logic as inference.py)
            for i in range(n_sample):
                if i % 10 == digit:
                    img = (x_gen[i] + 1) / 2
                    digit_images.append(img)
                    
                    # Save first "0" and first "1" for reverse process visualization
                    if digit == 0 and first_zero is None:
                        first_zero = img.clone()
                    elif digit == 1 and first_one is None:
                        first_one = img.clone()
                    
                    if len(digit_images) == 10:
                        break
            
            grid_images.extend(digit_images)
        
        # Create grid: 10 rows x 10 columns
        grid_images = torch.stack(grid_images)
        grid = make_grid(grid_images, nrow=10, normalize=False)
        save_image(grid, os.path.join(save_dir, 'generated_images_grid.png'))
        
        print(f"Generated 100 images grid (10x10)")
        return first_zero, first_one

def visualize_reverse_process(save_dir, first_zero, first_one, model_path):
    """Visualize the reverse process (denoising) for the first 0 and first 1"""
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=256, n_classes=10),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.0,
    )
    ddpm.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ddpm.to(device)

    os.makedirs(save_dir, exist_ok=True)

    # Time steps to visualize: t=1000 (noisiest) to t=0 (clearest)
    # Will be displayed left to right: 1000, 800, 600, 400, 200, 0
    target_times = [1000, 800, 600, 400, 200, 0]
    
    ddpm.eval()
    with torch.no_grad():
        for digit, target_img in [(0, first_zero), (1, first_one)]:
            reverse_imgs_dict = {t: None for t in target_times}
            
            # Start from pure noise
            x_t = torch.randn((1, 3, 28, 28), device=device)
            c = torch.tensor([digit], device=device)
            
            # Capture initial pure noise at t=1000
            reverse_imgs_dict[1000] = (x_t.clone() + 1) / 2
            
            # Manually denoise step by step (from t=999 down to t=0)
            for t in range(n_T - 1, -1, -1):
                # Capture BEFORE denoising at specific timesteps (except t=1000 which we already captured)
                if t in target_times and t < 1000:
                    reverse_imgs_dict[t] = (x_t.clone() + 1) / 2
                
                if t > 0:  # Only denoise if not at the last step
                    t_tensor = torch.tensor([t / n_T], device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    t_tensor = t_tensor.repeat(1, 1, 1, 1)
                    context_mask = torch.tensor([0.0], device=device)
                    
                    # Get model prediction
                    with torch.autocast(device_type=device):
                        pred_noise = ddpm.nn_model(x_t, c, t_tensor, context_mask)
                    
                    # Denoise using the DDPM formula
                    z = torch.randn_like(x_t) if t > 1 else 0
                    
                    x_t = (
                        ddpm.oneover_sqrta[t] * (x_t - pred_noise * ddpm.mab_over_sqrtmab[t])
                        + ddpm.sqrt_beta_t[t] * z
                    )
            
            # Use the pre-generated image as the final clean image (t=0)
            reverse_imgs_dict[0] = target_img.unsqueeze(0)
            
            # Create grid visualization in order: t=1000, 800, 600, 400, 200, 0 (noisiest to clearest, left to right)
            reverse_imgs = [reverse_imgs_dict[t] for t in target_times]
            reverse_imgs = torch.cat(reverse_imgs, dim=0)
            grid = make_grid(reverse_imgs, nrow=6, normalize=False)
            save_image(grid, os.path.join(save_dir, f'reverse_process_digit_{digit}.png'))
            
            print(f"Saved reverse process visualization for digit {digit}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_image_dir', type=pathlib.Path, required=False, default='report_images/')
    parser.add_argument("--model_path", type=pathlib.Path, required=False, default='checkpoints/model_199.pth')
    args = parser.parse_args()

    set_seed(42)
    
    os.makedirs(args.output_image_dir, exist_ok=True)

    print("Starting image generation and reverse process visualization...")
    print(datetime.now())
    
    # Generate 100 images grid
    first_zero, first_one = generate_100_grid(args.output_image_dir, args.model_path)
    
    # Visualize reverse process
    visualize_reverse_process(args.output_image_dir, first_zero, first_one, args.model_path)
    
    print(datetime.now())
    print("Complete!")