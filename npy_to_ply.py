import numpy as np
import argparse
from plyfile import PlyData, PlyElement

def convert_to_ply(input_path, output_path):
    print(f"Loading {input_path}...")
    # Load the data
    data = np.load(input_path)
    
    # Check shape
    print(f"Original shape: {data.shape}")
    
    # Case 1: Dense Map (H, W, 3) -> Flatten to (N, 3)
    if data.ndim == 3 and data.shape[2] == 3:
        points = data.reshape(-1, 3)
    # Case 2: Already a list of points (N, 3)
    elif data.ndim == 2 and data.shape[1] == 3:
        points = data
    else:
        raise ValueError(f"Unsupported shape {data.shape}. Expected (H,W,3) or (N,3).")

    # Filter out invalid points (optional but recommended)
    # Remove points at exactly (0,0,0) or NaNs if DUSt3R uses them for background
    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    # Optional: Filter exact zeros if your data uses (0,0,0) as void
    # valid_mask &= (np.abs(points).sum(axis=1) > 1e-6) 
    
    points = points[valid_mask]
    print(f"Valid points: {len(points)}")

    # Create structured array for PlyWriter
    vertex = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPY point cloud to PLY")
    parser.add_argument("input", help="Path to input .npy file")
    parser.add_argument("output", help="Path to output .ply file")
    args = parser.parse_args()
    
    convert_to_ply(args.input, args.output)