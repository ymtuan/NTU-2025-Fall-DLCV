# render_p2.py
import torch
import os
from os import makedirs
from argparse import ArgumentParser
from tqdm import tqdm
import torchvision
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import GaussianModel, render
from utils.general_utils import safe_state

# --- ADDED THIS IMPORT ---
from utils.pose_utils import get_tensor_from_camera 

def render_submission(dataset, iteration, pipeline, args):
    with torch.no_grad():
        # 1. Load the Gaussian Model
        gaussians = GaussianModel(dataset.sh_degree)
        
        # Force eval to True so Scene loads the test set (sparse_3/1)
        dataset.eval = True
        
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 2. Prepare Output Directory
        output_dir = args.output_path
        makedirs(output_dir, exist_ok=True)

        # 3. Get Test Cameras
        test_cameras = scene.getTestCameras()
        
        print(f"Found {len(test_cameras)} test images.")
        print(f"Rendering to {output_dir}...")

        # 4. Render Loop
        for view in tqdm(test_cameras, desc="Rendering"):
            # --- ADDED POSE CALCULATION HERE ---
            # InstantSplat requires explicit camera pose tensor
            camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
            
            # Pass camera_pose to the render function
            rendering = render(
                view, 
                gaussians, 
                pipeline, 
                background, 
                camera_pose=camera_pose
            )["render"]
            
            # Save Image
            save_name = view.image_name + ".png"
            torchvision.utils.save_image(
                rendering, 
                os.path.join(output_dir, save_name)
            )

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_path", type=str, required=True)
    
    # Do NOT add --eval here to avoid conflict with ModelParams
    
    args = get_combined_args(parser)
    
    # Manually force eval to True
    args.eval = True 
    
    print(f"Loading model from: {args.model_path}")
    safe_state(args.quiet)
    
    render_submission(model.extract(args), args.iteration, pipeline.extract(args), args)