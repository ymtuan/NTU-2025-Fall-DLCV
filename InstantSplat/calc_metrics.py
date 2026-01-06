import os
import shutil
import subprocess
import sys
import glob

def get_rendered_images(output_root):
    # Find where the renders are stored. Usually outputs/test/ours_30000/renders or similar
    # We search recursively for a "renders" folder inside the output dir
    render_dirs = glob.glob(os.path.join(output_root, "**", "renders"), recursive=True)
    if not render_dirs:
        return []
    
    # Use the first valid render folder found
    render_dir = render_dirs[0]
    print(f"[Metrics] Found rendered images in: {render_dir}")
    
    # List all png files
    images = [f for f in os.listdir(render_dir) if f.endswith(".png")]
    return images

def sanitize_for_metrics(sparse_folder, allowed_images_list):
    """
    Rewrites images.txt to ONLY include images that are in the 'allowed_images_list'.
    This ensures metrics.py only calculates scores for the images you actually rendered.
    """
    # Fix images.txt
    img_file = os.path.join(sparse_folder, "images.txt")
    if os.path.exists(img_file):
        with open(img_file, 'r') as f:
            lines = f.readlines()
        
        header = [l for l in lines if l.strip().startswith('#') and 'Number of' not in l]
        
        valid_pairs = []
        current_meta = None
        
        for line in lines:
            sline = line.strip()
            if sline.startswith('#'): continue
            
            if current_meta is None:
                if sline: current_meta = line
            else:
                # Check if this image is in our allowed list
                parts = current_meta.split()
                if len(parts) >= 10:
                    image_name = parts[9]
                    if image_name in allowed_images_list:
                        valid_pairs.append((current_meta, line))
                current_meta = None
        
        real_count = len(valid_pairs)
        print(f"[{sparse_folder}] Filtered for metrics: {real_count} images (matching your renders).")
        
        with open(img_file, 'w') as f:
            for h in header: f.write(h)
            f.write(f"# Number of images: {real_count}, mean observations per image: 0.0\n")
            for meta, points in valid_pairs:
                f.write(meta)
                f.write(points)
    
    # Fix cameras.txt (Just simple header fix)
    cam_file = os.path.join(sparse_folder, "cameras.txt")
    if os.path.exists(cam_file):
        with open(cam_file, 'r') as f:
            lines = f.readlines()
        header = [l for l in lines if l.strip().startswith('#') and 'Number of' not in l]
        data = [l for l in lines if not l.strip().startswith('#') and l.strip()]
        with open(cam_file, 'w') as f:
            for h in header: f.write(h)
            f.write(f"# Number of cameras: {len(data)}\n")
            for d in data: f.write(d)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 calc_metrics_safe.py <data_root> <output_dir>")
        sys.exit(1)

    original_data = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    temp_data = os.path.abspath("temp_metrics_work")
    n_views = "3"

    # 1. Detect rendered images
    rendered_images = get_rendered_images(output_dir)
    if not rendered_images:
        print("Error: Could not find any 'renders' folder in your output directory.")
        sys.exit(1)
    
    print(f"[Metrics] Target images: {rendered_images}")

    # 2. Setup Temp Data
    print(f"[Metrics] Setting up temp environment in {temp_data}...")
    if os.path.exists(temp_data):
        shutil.rmtree(temp_data)
    os.makedirs(temp_data)

    try:
        os.symlink(os.path.join(original_data, "images"), os.path.join(temp_data, "images"))
    except FileExistsError:
        pass
    shutil.copytree(os.path.join(original_data, "sparse_3"), os.path.join(temp_data, "sparse_3"))

    # 3. Sanitize (Strict filtering based on renders)
    # We modify the 'test' split (1) to only contain the renders we have.
    sanitize_for_metrics(os.path.join(temp_data, "sparse_3", "1"), rendered_images)
    # We also fix the train split (0) just to prevent loader crashes, though metrics.py mainly checks test
    sanitize_for_metrics(os.path.join(temp_data, "sparse_3", "0"), rendered_images) 

    # 4. Run Metrics
    print("[Metrics] Running official metrics.py...")
    cmd_metrics = [
        "python3", "metrics.py",
        "-s", temp_data,
        "-m", output_dir,
        "--n_views", n_views
    ]
    subprocess.run(cmd_metrics, check=True)

    # 5. Cleanup
    print("[Metrics] Cleaning up...")
    if os.path.exists(temp_data):
        shutil.rmtree(temp_data)
    print("Done.")

if __name__ == "__main__":
    main()