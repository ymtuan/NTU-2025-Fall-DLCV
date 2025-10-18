import torch
from model import DDIM
from UNet import UNet
from torchvision.utils import save_image
import os
import argparse
import pathlib

def gen_image(noise_path, model, output_dir):

    channels = 3
    height = 256
    width = 256

    # Load the noise tensor
    ground_truth_noise = torch.load(noise_path, map_location=device)
    print(f"Loaded noise from {noise_path}]")
    if ground_truth_noise.shape != (1, channels, height, width):
        print(f"Warning: Noise shape {ground_truth_noise.shape} does not match expected {(1, channels, height, width)}")

    # Normalize noise to [-1, 1] based on observed range
    noise_min = ground_truth_noise.min()
    noise_max = ground_truth_noise.max()
    if noise_max > 1.0 or noise_min < -1.0:
        ground_truth_noise = 2 * (ground_truth_noise - noise_min) / (noise_max - noise_min) - 1
        print(f"Normalized noise to range [-1, 1] from [{noise_min}, {noise_max}]")
    
    # Perform DDIM Sampling to generate image
    generated_image = model.sample(
        batch_size=1,
        channels=channels,
        height=height,
        width=width,
        ground_truth_noise=ground_truth_noise,
        save_intermediate=False
    )

    # Save the generated image
    min_gen = torch.min(generated_image)
    max_gen = torch.max(generated_image)
    norm_generated_image = (generated_image - min_gen) / (max_gen - min_gen)
    output_image_path = os.path.join(output_dir, f"{os.path.basename(noise_path).split('.')[0]}.png")
    save_image(norm_generated_image, output_image_path)
    print(f"Saved output image as {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_dir', type=pathlib.Path, required=False, default='../../hw2_data/face/noise')
    parser.add_argument("--output_dir", type=pathlib.Path, required=False, default='output')
    parser.add_argument('--unet_model_path', type=pathlib.Path, required=False, default='../../hw2_data/face/UNet.pt')
    args = parser.parse_args()

    eta = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unet_model = UNet().to(device)
    state_dict = torch.load(args.unet_model_path, map_location=device)
    unet_model.load_state_dict(state_dict)
    unet_model.eval()
    print(f"Loaded pre-trained UNet model from {args.unet_model_path}")

    DDIM_model = DDIM(
        model=unet_model,
        n_timesteps=1000,
        n_steps=50,
        eta=eta,
        device=device
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(10):
        noise_path = os.path.join(args.noise_dir, f"{i:02d}.pt")
        if not os.path.exists(noise_path):
            print(f"Noise file {noise_path} does not exist. Skipping.")
            continue

        gen_image(noise_path=noise_path, model=DDIM_model, output_dir=args.output_dir)