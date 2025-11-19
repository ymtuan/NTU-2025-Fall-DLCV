#!/usr/bin/env python3
"""
ControlNet Inference Script
Reads prompt.json and generates images with ControlNet conditioning
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add ControlNet to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cldm.model import load_state_dict, create_model
from cldm.ddim_hacked import DDIMSampler


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_image(image_path, target_size=512):
    """Load and resize image to target size"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return image


def save_image(x_tensor, filepath):
    """Save tensor as image"""
    x_sample = torch.clamp((x_tensor + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = (x_sample[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    output_image = Image.fromarray(x_sample)
    output_image.save(filepath)


def parse_args():
    parser = argparse.ArgumentParser(description="ControlNet Inference")
    parser.add_argument("--json_path", type=str, default="../../../hw2_data/fill50k/testing/prompt.json",
                        help="Path to prompt.json file")
    parser.add_argument("--input_dir", type=str, default="../../../hw2_data/fill50k/testing/source/",
                        help="Path to source images directory")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Path to output directory")
    parser.add_argument("--model_ckpt", type=str, default="lightning_logs/version_3/checkpoints/epoch=2-step=37424.ckpt",
                        help="Path to ControlNet model checkpoint (your trained weights)")
    parser.add_argument("--sd_model", type=str, default="../../../stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt",
                        help="Path to Stable Diffusion base checkpoint (TA-provided)")
    parser.add_argument("--config", type=str, default="models/cldm_v15.yaml",
                        help="Path to model config")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of images to generate per prompt")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta (0 for deterministic)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device and seed
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    print(f"Device: {device}")
    # Load model
    model = create_model(args.config).to(device)

    # First load SD base (TA-provided) with strict=False
    if args.sd_model and os.path.exists(args.sd_model):
        print(f"Loading SD base from {args.sd_model}")
        sd_base = load_state_dict(args.sd_model, location=device)
        _m_sd, _u_sd = model.load_state_dict(sd_base, strict=False)
    else:
        print(f"Warning: SD base checkpoint not found: {args.sd_model}")

    # Then load your ControlNet weights to fill control_model.* (strict=False)
    if not os.path.exists(args.model_ckpt):
        print(f"Error: ControlNet checkpoint not found: {args.model_ckpt}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading ControlNet from {args.model_ckpt}")
    load_dict = load_state_dict(args.model_ckpt, location=device)
    m, u = model.load_state_dict(load_dict, strict=False)

    if len(m) > 0:
        print(f"Missing keys: {m}")
    if len(u) > 0:
        print(f"Unexpected keys: {u}")

    model = model.eval()
    
    # Create DDIM sampler
    ddim_sampler = DDIMSampler(model)
    
    # Read prompt.json
    print(f"\nReading prompts from {args.json_path}")
    prompts_data = []
    with open(args.json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts_data.append(json.loads(line))
    
    print(f"Found {len(prompts_data)} prompts")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each prompt
    for idx, item in enumerate(tqdm(prompts_data, desc="Generating images")):
        source_filename = item['source']
        prompt = item['prompt']
        target_filename = item['target']
        
        # Load control image
        control_image_path = os.path.join(args.input_dir, source_filename)
        if not os.path.exists(control_image_path):
            print(f"Warning: Source image not found: {control_image_path}")
            continue
        
        control_image = load_image(control_image_path).to(device)
        
        # Encode text prompt to conditioning
        with torch.no_grad():
            cond = model.get_learned_conditioning([prompt])
            un_cond = model.get_learned_conditioning([""])
        
        # Generate one sample per prompt
        # Set seed for reproducibility
        set_seed(args.seed + idx)
        
        with torch.no_grad():
            # Prepare conditioning dict
            cond_dict = {
                "c_crossattn": [cond],
                "c_concat": [control_image]
            }
            un_cond_dict = {
                "c_crossattn": [un_cond],
                "c_concat": [control_image]
            }
            
            # Generate sample
            samples, _ = ddim_sampler.sample(
                S=args.num_steps,
                conditioning=cond_dict,
                batch_size=1,
                shape=[4, 64, 64],
                verbose=False,
                unconditional_guidance_scale=args.guidance_scale,
                unconditional_conditioning=un_cond_dict,
                eta=args.eta,
                x_T=torch.randn([1, 4, 64, 64], device=device)
            )
            
            # Decode from latent space
            x_samples = model.decode_first_stage(samples)
            
            # Save image directly to output_dir with condition name
            output_path = output_dir / f"{target_filename}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(x_samples, output_path)
    
    print(f"\nInference complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()