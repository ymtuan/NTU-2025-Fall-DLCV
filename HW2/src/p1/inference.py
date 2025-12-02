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
from model import DDPM, ContextUnet  # Import from model.py
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

def output_images(save_dir, model_path):

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

    # Generate for all digits using the single model
    images_path = save_dir
    os.makedirs(images_path, exist_ok=True)
    mnistm_path = os.path.join(images_path, 'mnistm')
    os.makedirs(mnistm_path, exist_ok=True)
    svhn_path = os.path.join(images_path, 'svhn')
    os.makedirs(svhn_path, exist_ok=True)

    ddpm.eval()
    with torch.no_grad():
        n_sample = 50 * n_classes
        torch.manual_seed(42)
        with torch.autocast(device_type=device):
            x_gen, _ = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=2.0)

        for i in range(n_sample):
            label = i % 10
            folder = mnistm_path if label % 2 == 0 else svhn_path
            filename = f"{label}_{int((i - label)/10)+1:03d}.png"
            save_image((x_gen[i] + 1) / 2, os.path.join(folder, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # should change to required path
    parser.add_argument('--output_image_dir', type=pathlib.Path, required=False, default='output')
    parser.add_argument("--model_path", type=pathlib.Path, required=False, default='checkpoints/model_199.pth')
    args = parser.parse_args()

    set_seed(42)
    
    os.makedirs(args.output_image_dir, exist_ok=True)

    print(datetime.now())
    output_images(args.output_image_dir, args.model_path)
    print(datetime.now())