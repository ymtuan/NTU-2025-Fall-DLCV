import os
import shutil
import subprocess
import sys

def sanitize_file(sparse_folder, images_folder):
    """
    Reads colmap files and cleans them.
    Crucially, it checks if the image referenced in images.txt ACTUALLY EXISTS.
    """
    # --- 1. Fix cameras.txt ---
    cam_file = os.path.join(sparse_folder, "cameras.txt")
    if os.path.exists(cam_file):
        with open(cam_file, 'r') as f:
            lines = f.readlines()
        
        header = [l for l in lines if l.strip().startswith('#') and 'Number of' not in l]
        data = [l for l in lines if not l.strip().startswith('#') and l.strip()]
        
        # We blindly trust cameras.txt count for now (usually matches images)
        # But for safety, we just rewrite it with the correct line count.
        real_count = len(data)
        
        with open(cam_file, 'w') as f:
            for h in header: f.write(h)
            f.write(f"# Number of cameras: {real_count}\n")
            for d in data: f.write(d)

    # --- 2. Fix images.txt (Filter by Existence) ---
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
                if sline: 
                    current_meta = line
            else:
                # We have a pair: (Metadata Line, Points Line)
                # Parse the Metadata Line to get the filename
                # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                parts = current_meta.split()
                if len(parts) >= 10:
                    image_name = parts[9] # The last element is the filename
                    
                    # CHECK IF FILE EXISTS
                    target_img_path = os.path.join(images_folder, image_name)
                    if os.path.exists(target_img_path):
                        valid_pairs.append((current_meta, line))
                    else:
                        # Debug print (optional)
                        # print(f"Skipping missing image: {image_name}")
                        pass
                current_meta = None
        
        real_count = len(valid_pairs)
        print(f"[{sparse_folder}] Filtered images.txt: {real_count} valid images found (files exist).")
        
        with open(img_file, 'w') as f:
            for h in header: f.write(h)
            f.write(f"# Number of images: {real_count}, mean observations per image: 0.0\n")
            for meta, points in valid_pairs:
                f.write(meta)
                f.write(points)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 run_pipeline.py <data_root> <output_dir>")
        sys.exit(1)

    original_data = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    temp_data = os.path.abspath("temp_data_work")
    
    # Configuration
    n_views = "3"
    iterations = "1000"

    print(f"[Pipeline] Setting up temp environment in {temp_data}...")

    # --- SETUP TEMP FOLDER ---
    if os.path.exists(temp_data):
        shutil.rmtree(temp_data)
    os.makedirs(temp_data)

    try:
        os.symlink(os.path.join(original_data, "images"), os.path.join(temp_data, "images"))
    except FileExistsError:
        pass

    shutil.copytree(os.path.join(original_data, "sparse_3"), os.path.join(temp_data, "sparse_3"))

    # --- SANITIZE DATA (With Image Check) ---
    print("[Pipeline] Sanitizing headers in temp copy...")
    # Pass the images folder path so we can check existence
    images_dir = os.path.join(temp_data, "images")
    sanitize_file(os.path.join(temp_data, "sparse_3", "0"), images_dir)
    sanitize_file(os.path.join(temp_data, "sparse_3", "1"), images_dir)

    # --- RUN TRAINING ---
    print("[Pipeline] Starting Training...")
    cmd_train = [
        "python3", "train.py", "-s", temp_data, "-m", output_dir,
        "-r", "1", "--n_views", n_views, "--iterations", iterations,
        "--pp_optimizer", "--optim_pose"
    ]
    subprocess.run(cmd_train, check=True)

    # --- RUN RENDERING ---
    print("[Pipeline] Starting Rendering (Eval)...")
    cmd_render = [
        "python3", "render.py", "-s", temp_data, "-m", output_dir,
        "-r", "1", "--n_views", n_views, "--iterations", iterations,
        "--eval"
    ]
    subprocess.run(cmd_render, check=True)

    # --- CLEANUP ---
    print("[Pipeline] Cleaning up temp data...")
    if os.path.exists(temp_data):
        shutil.rmtree(temp_data)
    
    print("[Pipeline] Done.")

if __name__ == "__main__":
    main()