from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.transforms as trns
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from PIL import Image
import csv
from model import DDPM, ContextUnet  # Import from model.py
from dataset import DigitStyleDataset  # Import from dataset.py
from utils import EMA  # Import from utils.py

def train():
    # hardcoding these here
    n_epoch = 200
    batch_size = 128
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = 10
    n_feat = 256
    lrate = 2e-4
    save_model = True
    save_dir = "checkpoints/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance

    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.2,
    )
    ddpm.to(device)

    dataset = DigitStyleDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f"epoch {ep}")
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            optim.step()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        if ep == int(n_epoch - 1) or ep % 20 == 0:
            with torch.no_grad():
                n_sample = 4 * n_classes
                for w_i, w in enumerate(ws_test):
                    x_gen, x_gen_store = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=w)
                    grid = make_grid(x_gen * -1 + 1, nrow=10)
                    save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                    print("saved image at " + save_dir + f"image_ep{ep}_w{w}.png")

        # optionally save model
        if save_model and ep == int(n_epoch - 1) or ep % 20 == 0:
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print("saved model at " + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train()