import torch
import os

# Path to the noise folder
noise_folder = "../../hw2_data/face/noise"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Check the first noise file
for i in range(0, 10):
    noise_path = os.path.join(noise_folder, f"{i:02d}.pt")
    if os.path.exists(noise_path):
        ground_truth_noise = torch.load(noise_path, map_location=device)
        print(f"Noise file: {noise_path}")
        print(f"Shape: {ground_truth_noise.shape}")
        print(f"Min value: {ground_truth_noise.min()}")
        print(f"Max value: {ground_truth_noise.max()}")
        print(f"Mean value: {ground_truth_noise.mean()}")
        print(f"Data type: {ground_truth_noise.dtype}")
    else:
        print(f"Noise file {noise_path} does not exist.")