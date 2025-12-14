import os
import json
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import pycocotools.mask as mask_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True

class InsideDataset(Dataset):
    def __init__(self, json_path, image_dir, depth_dir=None, resize=(360, 640), 
                 use_depth=True, use_geometry=False, max_samples=None):
        with open(json_path, 'r') as f:
            all_data = json.load(f)
        
        # If max_samples is specified, sample balanced data
        if max_samples is not None and max_samples < len(all_data):
            # Separate positive and negative samples
            positive_samples = [d for d in all_data if d['inside'] == 1]
            negative_samples = [d for d in all_data if d['inside'] == 0]
            
            # Calculate how many samples per class
            samples_per_class = max_samples // 2
            
            # Sample from each class
            import random
            random.seed(42)  # For reproducibility
            
            if len(positive_samples) >= samples_per_class:
                sampled_positive = random.sample(positive_samples, samples_per_class)
            else:
                sampled_positive = positive_samples
                print(f"Warning: Only {len(positive_samples)} positive samples available, requested {samples_per_class}")
            
            if len(negative_samples) >= samples_per_class:
                sampled_negative = random.sample(negative_samples, samples_per_class)
            else:
                sampled_negative = negative_samples
                print(f"Warning: Only {len(negative_samples)} negative samples available, requested {samples_per_class}")
            
            self.data = sampled_positive + sampled_negative
            random.shuffle(self.data)  # Shuffle the combined data
            
            print(f"Dataset: {len(self.data)} samples (Positive: {len(sampled_positive)}, Negative: {len(sampled_negative)})")
        else:
            self.data = all_data
            pos_count = sum(1 for d in self.data if d['inside'] == 1)
            neg_count = len(self.data) - pos_count
            print(f"Dataset: {len(self.data)} samples (Positive: {pos_count}, Negative: {neg_count})")
        
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.use_depth = use_depth
        self.use_geometry = use_geometry
        self.resize = resize
        self.to_tensor = transforms.ToTensor()
        self.resize_tf = transforms.Resize(resize, interpolation=Image.BILINEAR)
        
    def __len__(self):
        return len(self.data)
    
    def decode_rle(self, rle):
        mask = mask_utils.decode(rle).astype(np.float32)
        return mask
    
    def compute_geometric_features(self, obj_mask, buffer_mask, depth_obj=None, depth_buffer=None):
        """
        Compute geometric features for the dual-stream model.
        Returns a tensor of shape [8] with the following features:
        1. IoU (Intersection over Union)
        2. Object Area / Image Area
        3. Buffer Area / Image Area
        4. Overlap Area / Object Area
        5. Overlap Area / Buffer Area
        6. Object Center X (normalized)
        7. Object Center Y (normalized)
        8. Depth Difference (mean depth of object - mean depth of buffer)
        """
        H, W = obj_mask.shape
        img_area = H * W
        
        # Convert to numpy if needed
        if isinstance(obj_mask, torch.Tensor):
            obj_mask = obj_mask.numpy()
        if isinstance(buffer_mask, torch.Tensor):
            buffer_mask = buffer_mask.numpy()
        
        obj_mask_binary = (obj_mask > 0.5).astype(np.float32)
        buffer_mask_binary = (buffer_mask > 0.5).astype(np.float32)
        
        # Feature 1: IoU
        intersection = np.sum(obj_mask_binary * buffer_mask_binary)
        union = np.sum(np.clip(obj_mask_binary + buffer_mask_binary, 0, 1))
        iou = intersection / (union + 1e-8)
        
        # Feature 2-3: Normalized areas
        obj_area = np.sum(obj_mask_binary)
        buffer_area = np.sum(buffer_mask_binary)
        obj_area_norm = obj_area / img_area
        buffer_area_norm = buffer_area / img_area
        
        # Feature 4-5: Overlap ratios
        overlap_obj_ratio = intersection / (obj_area + 1e-8)
        overlap_buffer_ratio = intersection / (buffer_area + 1e-8)
        
        # Feature 6-7: Object center (normalized to [0, 1])
        obj_coords = np.where(obj_mask_binary > 0)
        if len(obj_coords[0]) > 0:
            center_y = np.mean(obj_coords[0]) / H
            center_x = np.mean(obj_coords[1]) / W
        else:
            center_x, center_y = 0.5, 0.5
        
        # Feature 8: Depth difference
        if depth_obj is not None and depth_buffer is not None:
            if isinstance(depth_obj, torch.Tensor):
                depth_obj = depth_obj.numpy()
            if isinstance(depth_buffer, torch.Tensor):
                depth_buffer = depth_buffer.numpy()
            
            obj_depth_mean = np.mean(depth_obj[obj_mask_binary > 0]) if np.sum(obj_mask_binary) > 0 else 0
            buffer_depth_mean = np.mean(depth_buffer[buffer_mask_binary > 0]) if np.sum(buffer_mask_binary) > 0 else 0
            depth_diff = obj_depth_mean - buffer_depth_mean
        else:
            depth_diff = 0.0
        
        features = np.array([
            iou,
            obj_area_norm,
            buffer_area_norm,
            overlap_obj_ratio,
            overlap_buffer_ratio,
            center_x,
            center_y,
            depth_diff
        ], dtype=np.float32)
        
        return torch.from_numpy(features)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_dir, item['image'])
        img = Image.open(img_path).convert('RGB')
        img = self.resize_tf(img)
        img = self.to_tensor(img)  # 3 x H x W
        
        # Optional: Depth
        depth = None
        if self.use_depth and self.depth_dir is not None:
            depth_path = os.path.join(self.depth_dir, item['image'].replace('.png', '_depth.png'))
            if os.path.exists(depth_path):
                depth = Image.open(depth_path)
                depth = self.resize_tf(depth)
                depth = self.to_tensor(depth)  # 1 x H x W
            else:
                raise FileNotFoundError(f"Depth file not found: {depth_path}")

        # Masks
        buffer_mask = self.decode_rle(item['buffer_rle'])  # (H, W)
        obj_mask = self.decode_rle(item['obj_rle'])        # (H, W)

        # Resize masks to target shape
        buffer_mask = Image.fromarray((buffer_mask * 255).astype(np.uint8))
        buffer_mask = self.resize_tf(buffer_mask)
        buffer_mask = self.to_tensor(buffer_mask)  # 1 x H x W

        obj_mask = Image.fromarray((obj_mask * 255).astype(np.uint8))
        obj_mask = self.resize_tf(obj_mask)
        obj_mask = self.to_tensor(obj_mask)  # 1 x H x W

        # Stack input channels
        if self.use_depth:
            x = torch.cat([img, depth, buffer_mask, obj_mask], dim=0)
        else:
            x = torch.cat([img, buffer_mask, obj_mask], dim=0)
        
        y = torch.tensor(item['inside'], dtype=torch.float32)

        # Compute geometric features if needed
        if self.use_geometry:
            depth_obj = depth.squeeze(0) if depth is not None else None
            depth_buffer = depth.squeeze(0) if depth is not None else None
            geo_features = self.compute_geometric_features(
                obj_mask.squeeze(0), 
                buffer_mask.squeeze(0),
                depth_obj,
                depth_buffer
            )
            return x, geo_features, y
        else:
            return x, y