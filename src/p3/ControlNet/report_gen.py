#!/usr/bin/env python3
"""
ControlNet Circle Test - Two simple sets with white border circles on black background
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import subprocess
import sys


def create_circle_control_image(circles, size=512, bg_color=(0, 0, 0)):
    """
    Create a control image with circles (white border, black fill) on black background
    
    Args:
        circles: list of dicts with keys 'center', 'radius', 'border_width'
        size: image size
        bg_color: background color (R, G, B)
    
    Returns:
        PIL Image
    """
    img = Image.new('RGB', (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    for circle in circles:
        cx, cy = circle['center']
        r = circle['radius']
        border_width = circle.get('border_width', 3)
        
        # Draw circle with white border and black fill
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(0, 0, 0), outline=(255, 255, 255), width=border_width)
    
    return img


def create_test_data(output_dir):
    """Create two simple test circle images and prompt.json"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set 1: Two circles with white borders
    set1_circles = [
        {'center': (101, 152), 'radius': 60, 'border_width': 3},
        {'center': (342, 277), 'radius': 60, 'border_width': 3}
    ]
    img1 = create_circle_control_image(set1_circles)
    img1_path = output_dir / "circles_set1.png"
    img1.save(img1_path)
    print(f"Created Set 1: {img1_path}")
    
    # Set 2: Two circles with white borders
    set2_circles = [
        {'center': (150, 106), 'radius': 60, 'border_width': 3},
        {'center': (256, 306), 'radius': 60, 'border_width': 3}
    ]
    img2 = create_circle_control_image(set2_circles)
    img2_path = output_dir / "circles_set2.png"
    img2.save(img2_path)
    print(f"Created Set 2: {img2_path}")
    
    # Create prompt.json with two entries
    prompts = [
        {
            "source": "circles_set1.png",
            "target": "output_set1.png",
            "prompt": "a red circle and a blue circle with pink background"
        },
        {
            "source": "circles_set2.png",
            "target": "output_set2.png",
            "prompt": "a green circle and a yellow circle with blue background"
        }
    ]
    
    prompt_path = output_dir / "test_circles_prompt.json"
    with open(prompt_path, 'w') as f:
        for item in prompts:
            f.write(json.dumps(item) + '\n')
    print(f"Created: {prompt_path}")
    
    return img1_path, img2_path, prompt_path


def run_inference(json_path, input_dir, output_dir, model_ckpt, config, 
                  num_steps=50, guidance_scale=7.5):
    """Run ControlNet inference"""
    
    inference_script = Path(__file__).parent / "inference.py"
    
    if not inference_script.exists():
        print(f"Error: inference.py not found at {inference_script}")
        print("Make sure you run this script from the ControlNet directory")
        return False
    
    cmd = [
        "python3", str(inference_script),
        "--json_path", str(json_path),
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--model_ckpt", str(model_ckpt),
        "--config", str(config),
        "--num_steps", str(num_steps),
        "--guidance_scale", str(guidance_scale),
        "--seed", "42"
    ]
    
    print("\n" + "="*60)
    print("Running ControlNet Inference")
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    print("="*60 + "\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate two simple circle control images and test ControlNet"
    )
    parser.add_argument("--test_data_dir", type=str, default="./test_circles_simple",
                        help="Directory to save circle images and prompts")
    parser.add_argument("--output_dir", type=str, default="./circles_output_simple",
                        help="Directory to save generated images")
    parser.add_argument("--model_ckpt", type=str, 
                        default="lightning_logs/version_3/checkpoints/epoch=2-step=37424.ckpt",
                        help="Path to trained ControlNet checkpoint")
    parser.add_argument("--config", type=str, default="models/cldm_v15.yaml",
                        help="Path to model config")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Only create test images, skip inference")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ControlNet Circle Test - Two Simple Sets")
    print("="*60 + "\n")
    
    # Step 1: Create test data
    print("Step 1: Creating simple circle images...")
    print("-" * 60)
    os.makedirs(args.test_data_dir, exist_ok=True)
    img1, img2, prompt_path = create_test_data(args.test_data_dir)
    print(f"\nTest data created in: {args.test_data_dir}")
    print(f"Set 1: Two circles with white borders on black background")
    print(f"Set 2: Two circles with white borders on black background")
    print(f"Prompts file: {prompt_path}")
    
    # Step 2: Run inference
    if not args.skip_inference:
        print("\n\nStep 2: Running ControlNet inference...")
        print("-" * 60)
        
        success = run_inference(
            json_path=prompt_path,
            input_dir=args.test_data_dir,
            output_dir=args.output_dir,
            model_ckpt=args.model_ckpt,
            config=args.config,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale
        )
        
        if success:
            print("\n" + "="*60)
            print("SUCCESS! Generated images saved to:")
            print(f"  {args.output_dir}/")
            print(f"  - output_set1.png")
            print(f"  - output_set2.png")
            print("="*60)
            print("\nNext steps:")
            print("1. Compare control images with generated images")
            print("2. Assess circle shape, position, and color preservation")
            print("3. Include both sets in your homework report (Section 3, Part 2)")
            print("="*60 + "\n")
        else:
            print("\nError: Inference failed. Check the error messages above.")
            return 1
    else:
        print("\n" + "="*60)
        print("Test images created. Inference skipped.")
        print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())