import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import torchvision.transforms.functional as F
import pycocotools.mask as mask_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DistanceDataset(Dataset):
    def __init__(self, data_dir, json_path, transform=None, rgb=True, depth=True, 
                 resize=(360, 640), cls_bin_center=None, distance_scale=1., 
                 min_distance=-1., max_distance=100000, use_geometry=False, max_depth=65535.0):
        """
        Distance Dataset with optional Geometry-Aware features.
        
        Args:
            use_geometry (bool): If True, compute geometric features and return 6-channel input.
            max_depth (float): Maximum depth value for normalization (default: 65535.0 for 16-bit PNG).
        """
        # distance_scale is used to scale to different units (e.g. m, cm, mm)
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, 'images')
        self.depth_dir = os.path.join(data_dir, 'depths')
        self.use_rgb = rgb
        self.use_depth = depth
        self.use_geometry = use_geometry
        self.resize = resize
        self.transform = transform
        self.max_depth = max_depth
        
        if cls_bin_center is not None:
            self.cls_bin_center = torch.tensor(cls_bin_center, dtype=torch.float32)
        else:
            self.cls_bin_center = None
        self.distance_scale = distance_scale
        with open(json_path, 'r') as f:
            all_samples = json.load(f)

        print(f"Number of samples in JSON: {len(all_samples)}")

        # Filter out samples with missing files
        valid_samples = []
        missing_count = 0
        for s in all_samples:
            img_path = os.path.join(self.rgb_dir, s['image'])
            depth_path = os.path.join(self.depth_dir, s['image'].replace('.png', '_depth.png'))
            
            # Check if RGB image exists (required)
            if not os.path.exists(img_path):
                missing_count += 1
                continue
            
            # Check if depth exists (if needed)
            if (self.use_depth or self.use_geometry) and not os.path.exists(depth_path):
                missing_count += 1
                continue
            
            valid_samples.append(s)
        
        if missing_count > 0:
            print(f"Warning: Skipped {missing_count} samples due to missing image/depth files")
        
        print(f"Valid samples after file check: {len(valid_samples)}")

        print(f"Apply data filtering: Min distance: {min_distance}, Max distance: {max_distance}")
        self.samples = [s for s in valid_samples if s['normalized_answer'] >= min_distance and s['normalized_answer'] <= max_distance]
        print(f"Number of samples after distance filtering: {len(self.samples)}")
        
        if self.use_geometry:
            print("✓ Geometry-Aware mode enabled: 6-channel input + geometric features")

    def __len__(self):
        return len(self.samples)

    def decode_mask(self, rle):
        # Handle both string and bytes format
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('utf-8')
        return mask_utils.decode(rle).astype(np.float32)
    
    def compute_geo_features(self, mask1, mask2, depth_map):
        """
        Compute hand-crafted geometric features for dual-stream architecture.
        
        Args:
            mask1: Binary mask (H, W) as numpy array
            mask2: Binary mask (H, W) as numpy array
            depth_map: Depth map (H, W) as numpy array, normalized to [0, 1]
        
        Returns:
            geo_features: Tensor of shape [12] containing:
                - centroid_dist_2d: Normalized 2D distance between centroids
                - depth_diff: Absolute depth difference between masks
                - mean_depth_1: Mean depth of mask1
                - mean_depth_2: Mean depth of mask2
                - area_1: Normalized sqrt(area) of mask1
                - area_2: Normalized sqrt(area) of mask2
                - bbox_coords: [x1, y1, x2, y2] for mask1 (4 values, normalized)
                - bbox_coords: [x1, y1, x2, y2] for mask2 (4 values, normalized)
        """
        H, W = mask1.shape
        diag = np.sqrt(H**2 + W**2)  # Image diagonal for normalization
        
        # Ensure binary masks
        mask1_bin = (mask1 > 0.5).astype(np.float32)
        mask2_bin = (mask2 > 0.5).astype(np.float32)
        
        features = []
        
        # --- Feature 1: Centroid distance (2D) ---
        coords1 = np.where(mask1_bin > 0)
        coords2 = np.where(mask2_bin > 0)
        
        if len(coords1[0]) > 0:
            centroid1 = np.array([np.mean(coords1[1]), np.mean(coords1[0])])  # (x, y)
        else:
            centroid1 = np.array([W/2, H/2])  # Default to image center
        
        if len(coords2[0]) > 0:
            centroid2 = np.array([np.mean(coords2[1]), np.mean(coords2[0])])  # (x, y)
        else:
            centroid2 = np.array([W/2, H/2])
        
        centroid_dist = np.linalg.norm(centroid1 - centroid2) / diag
        features.append(centroid_dist)
        
        # --- Features 2-4: Depth statistics ---
        if len(coords1[0]) > 0:
            depth1_vals = depth_map[coords1[0], coords1[1]]
            mean_depth_1 = np.mean(depth1_vals)
        else:
            mean_depth_1 = 0.0
        
        if len(coords2[0]) > 0:
            depth2_vals = depth_map[coords2[0], coords2[1]]
            mean_depth_2 = np.mean(depth2_vals)
        else:
            mean_depth_2 = 0.0
        
        depth_diff = abs(mean_depth_1 - mean_depth_2)
        
        features.append(depth_diff)
        features.append(mean_depth_1)
        features.append(mean_depth_2)
        
        # --- Features 5-6: Normalized areas ---
        area1 = np.sum(mask1_bin)
        area2 = np.sum(mask2_bin)
        
        area1_norm = np.sqrt(area1) / diag
        area2_norm = np.sqrt(area2) / diag
        
        features.append(area1_norm)
        features.append(area2_norm)
        
        # --- Features 7-10: Bounding box for mask1 ---
        if len(coords1[0]) > 0:
            y_min, y_max = np.min(coords1[0]), np.max(coords1[0])
            x_min, x_max = np.min(coords1[1]), np.max(coords1[1])
            bbox1 = [x_min / W, y_min / H, x_max / W, y_max / H]
        else:
            bbox1 = [0.0, 0.0, 0.0, 0.0]
        
        features.extend(bbox1)
        
        # --- Features 11-14: Bounding box for mask2 ---
        if len(coords2[0]) > 0:
            y_min, y_max = np.min(coords2[0]), np.max(coords2[0])
            x_min, x_max = np.min(coords2[1]), np.max(coords2[1])
            bbox2 = [x_min / W, y_min / H, x_max / W, y_max / H]
        else:
            bbox2 = [0.0, 0.0, 0.0, 0.0]
        
        features.extend(bbox2)
        
        return torch.tensor(features, dtype=torch.float32)

    def __getitem__(self, idx):
        item = self.samples[idx]
        components = []
        depth_map_np = None  # For geometric features

        # --- Load RGB ---
        if self.use_rgb:
            img_path = os.path.join(self.rgb_dir, item['image'])
            try:
                rgb = Image.open(img_path).convert('RGB')
                rgb = F.resize(rgb, self.resize)
                rgb = np.array(rgb).astype(np.float32) / 255.0
                components.append(rgb)
            except Exception as e:
                print(f"Warning: Failed to load RGB image {img_path}: {e}")
                return self.__getitem__((idx + 1) % len(self))

        # --- Load Depth (always load if use_geometry is True) ---
        if self.use_depth or self.use_geometry:
            depth_path = os.path.join(self.depth_dir, item['image'].replace('.png', '_depth.png'))
            try:
                depth = Image.open(depth_path)
                depth = F.resize(depth, self.resize, interpolation=Image.BILINEAR)
                depth_np = np.array(depth).astype(np.float32)
                
                # Normalize depth to [0, 1]
                depth_normalized = depth_np / self.max_depth
                depth_map_np = depth_normalized  # Save for geo features
                
                components.append(depth_normalized[..., None])
            except Exception as e:
                print(f"Warning: Failed to load depth image {depth_path}: {e}")
                # Use zero depth if not available
                depth_map_np = np.zeros(self.resize, dtype=np.float32)
                components.append(depth_map_np[..., None])

        # --- Load Masks ---
        try:
            mask_a = self.decode_mask(item['rle'][0])
            mask_b = self.decode_mask(item['rle'][1])
            mask_a = Image.fromarray(mask_a)
            mask_b = Image.fromarray(mask_b)
            mask_a = np.array(F.resize(mask_a, self.resize, interpolation=Image.NEAREST)).astype(np.float32)
            mask_b = np.array(F.resize(mask_b, self.resize, interpolation=Image.NEAREST)).astype(np.float32)
            components.append(mask_a[..., None])
            components.append(mask_b[..., None])
        except Exception as e:
            print(f"Warning: Failed to decode or resize masks: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # --- Stack input and convert to tensor ---
        input_tensor = np.concatenate(components, axis=-1)  # H x W x C (C=6 if RGB+Depth+2Masks)
        input_tensor = torch.tensor(input_tensor).permute(2, 0, 1)  # C x H x W
        
        # --- Compute geometric features if enabled ---
        geo_features = None
        if self.use_geometry:
            if depth_map_np is None:
                depth_map_np = np.zeros(self.resize, dtype=np.float32)
            geo_features = self.compute_geo_features(mask_a, mask_b, depth_map_np)
        
        # --- Process distance label ---
        distance_cls_idx = None
        if self.cls_bin_center is not None:
            distance_cls_idx = torch.argmin(torch.abs(self.cls_bin_center - item['normalized_answer']))
            distance = item['normalized_answer'] - self.cls_bin_center[distance_cls_idx] 
        else:
            distance = torch.tensor(item['normalized_answer']*self.distance_scale, dtype=torch.float32) 

        if self.transform:
            input_tensor = self.transform(input_tensor)

        # --- Return based on mode ---
        if self.use_geometry:
            if distance_cls_idx is not None:
                return input_tensor, geo_features, distance, distance_cls_idx
            else:
                return input_tensor, geo_features, distance
        else:
            if distance_cls_idx is not None:
                return input_tensor, distance, distance_cls_idx
            else:
                return input_tensor, distance