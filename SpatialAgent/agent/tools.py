import os
import sys
from mask import Mask
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from typing import List
import json
from mask import Mask, parse_masks_from_conversation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance_est.model import build_dist_model
from inside_pred.model import build_inside_model
from closest_pred.model import build_closest_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class tools_api:
    def __init__(self, dist_model_cfg, inside_model_cfg, small_dist_model_cfg=None, resize=(360,640), mask_IoU_thres=0.3, inside_thres=0.5, cascade_dist_thres=300, clamp_distance_thres=25, img_path=None, closest_dist_model_cfg=None):
        self.model = build_dist_model(dist_model_cfg)
        self.inside_model = build_inside_model(inside_model_cfg)
        # Build optional small distance model only if config is provided
        self.small_dist_model = build_dist_model(small_dist_model_cfg) if small_dist_model_cfg is not None else None
        
        # Build closest model - two modes:
        # 1. use_dedicated_closest=True: Use ClosenessScoreModel (outputs 0~1, higher=closer)
        # 2. use_dedicated_closest=False: Use distance model (outputs distance, lower=closer)
        if closest_dist_model_cfg is not None:
            self.use_dedicated_closest = closest_dist_model_cfg.get('use_dedicated_closest', False)
            if self.use_dedicated_closest:
                # Use dedicated closest model from closest_pred
                self.closest_model = build_closest_model(closest_dist_model_cfg)
                self.closest_num_geo_features = closest_dist_model_cfg.get('num_geo_features', 7)
            else:
                # Use distance model for closest (existing behavior)
                self.closest_model = build_dist_model(closest_dist_model_cfg)
            self.closest_use_geometry = closest_dist_model_cfg.get('use_geometry', False)
            self.closest_use_shortcut = closest_dist_model_cfg.get('use_shortcut', False)
        else:
            self.closest_model = self.model  # Fallback to dist model
            self.closest_use_geometry = dist_model_cfg.get('use_geometry', False)
            self.closest_use_shortcut = dist_model_cfg.get('use_shortcut', False)
            self.use_dedicated_closest = False
        self.use_geometry = inside_model_cfg.get('use_geometry', False)
        self.dist_use_geometry = dist_model_cfg.get('use_geometry', False)  # 記錄 distance model 是否使用 geometry
        self.dist_use_shortcut = dist_model_cfg.get('use_shortcut', False)  # 記錄是否使用 ResNetWithShortcut
        self.resize = resize
        self.mask_IoU_thres = mask_IoU_thres
        self.inside_thres = inside_thres
        self.cascade_dist_thres = cascade_dist_thres
        self.clamp_distance_thres = clamp_distance_thres
        self.img_path = img_path
        self.masks = None
        self.depth_path = None
    
    def update_masks(self, masks: List[Mask]):
        self.masks = masks

    def update_image(self, img_path):
        self.img_path = img_path
        assert os.path.exists(self.img_path), f"Image path {self.img_path} does not exist."
        # 如果 dist model、closest model 或 inside model 任一需要 geometry，就載入 depth
        if self.use_geometry or self.dist_use_geometry or self.closest_use_geometry:
            # 嘗試兩種 depth 路徑：
            # 1. 同目錄下的 _depth.png 後綴（舊格式）
            # 2. ../depths/ 目錄下的同名檔案（新格式）
            depth_path_1 = img_path.replace('.png', '_depth.png')
            
            # 將 /images/ 替換為 /depths/
            depth_path_2 = img_path.replace('/images/', '/depths/').replace('.png', '_depth.png')
            
            self.depth_path = depth_path_2

    def dist(self, mask_1: Mask, mask_2: Mask) -> float:
        print(f"\n[DEBUG dist()] dist_use_geometry={self.dist_use_geometry}, depth_path={self.depth_path}")

        # Check for significant mask overlap - if IoU is high enough, distance is 0
        mask1_array = mask_1.decode_mask()
        mask2_array = mask_2.decode_mask()
        
        # Compute IoU to detect significant overlap
        mask1_bin = mask1_array > 0
        mask2_bin = mask2_array > 0
        intersection = np.logical_and(mask1_bin, mask2_bin).sum()
        union = np.logical_or(mask1_bin, mask2_bin).sum()
        
        if union > 0:
            iou = intersection / union
            # Also check intersection ratio relative to smaller mask
            min_area = min(mask1_bin.sum(), mask2_bin.sum())
            intersection_ratio = intersection / min_area if min_area > 0 else 0
            
            # Only return 0 if very significant overlap (IoU > 0.3 or intersection covers >50% of smaller mask)
            if iou > 0.3 or intersection_ratio > 0.45:
                print(f"[DEBUG dist()] Significant overlap (IoU={iou:.3f}, int_ratio={intersection_ratio:.3f}), returning 0.0")
                return 0.0
            elif intersection > 0:
                print(f"[DEBUG dist()] Minor overlap (IoU={iou:.3f}, int_ratio={intersection_ratio:.3f}), proceeding with model")

        rgb = Image.open(self.img_path).convert('RGB')
        rgb = F.resize(rgb, self.resize)
        rgb = np.array(rgb).astype(np.float32) / 255.0

        # Resize masks (already decoded above for overlap check)
        mask1_img = Image.fromarray(mask1_array)
        mask2_img = Image.fromarray(mask2_array)
        mask1_img = F.resize(mask1_img, self.resize, interpolation=Image.NEAREST)
        mask2_img = F.resize(mask2_img, self.resize, interpolation=Image.NEAREST)
        mask1_resized = np.array(mask1_img).astype(np.float32)
        mask2_resized = np.array(mask2_img).astype(np.float32)

        # Load depth if using geometry
        depth_np = None
        if self.dist_use_geometry and self.depth_path and os.path.exists(self.depth_path):
            depth = Image.open(self.depth_path)
            depth = F.resize(depth, self.resize, interpolation=Image.BILINEAR)
            depth_np = np.array(depth).astype(np.float32) / 65535.0  # Normalize to [0, 1]
            
            # Stack inputs: RGB + Depth + 2 Masks = 6 channels
            components = [rgb, depth_np[..., None], mask1_resized[..., None], mask2_resized[..., None]]
        else:
            # Baseline: RGB + 2 Masks = 5 channels
            components = [rgb, mask1_resized[..., None], mask2_resized[..., None]]
            print(f"[DEBUG dist()] Using 5-channel input (RGB + 2 Masks)")
        
        input_tensor = np.concatenate(components, axis=-1)  # H x W x C
        input_tensor = torch.tensor(input_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # 1 x C x H x W

        # Compute geometric features if needed
        geo_features = None
        if self.dist_use_geometry and depth_np is not None:
            if self.dist_use_shortcut:
                # ResNetWithShortcut uses 3 simple geometric features
                geo_features = self._compute_simple_dist_geometric_features(mask1_resized, mask2_resized, depth_np)
                geo_features = geo_features.unsqueeze(0).to(DEVICE)  # [1, 3]
            else:
                # GeometryFusedResNet uses 14 geometric features
                geo_features = self._compute_dist_geometric_features(mask1_resized, mask2_resized, depth_np)
                geo_features = geo_features.unsqueeze(0).to(DEVICE)  # [1, 14]

        with torch.no_grad():
            if self.dist_use_geometry and geo_features is not None:
                # 使用 geometry model 時，只用主模型，不使用 cascade small model
                # 輸出已經是正確單位的距離，不需要除以 100
                #print(f"[DEBUG dist()] Calling model with geo_features")
                predicted_distance = self.model(input_tensor, geo_features).item()
                # Clamp negative distances to 0 (distance should be non-negative)
                predicted_distance = max(predicted_distance, 0.0)
                #print(f"[DEBUG dist()] Predicted distance (geometry model, no /100): {predicted_distance}")
            else:
                # 使用 baseline model 時，使用 cascade 策略
                # Baseline model 輸出是 distance×100，需要除以 100
                #print(f"[DEBUG dist()] Calling baseline model (will /100)")
                predicted_distance = self.model(input_tensor).item() / 100.0
                print(f"[DEBUG dist()] large scale Predicted distance (after /100): {predicted_distance}")
                
                # Baseline model 才使用 small model cascade（如果已提供 small model）
                if self.small_dist_model is not None and predicted_distance < self.cascade_dist_thres:
                    predicted_distance = self.small_dist_model(input_tensor).item() / 100.0
                    print(f"[DEBUG dist()] small scale Predicted distance (after /100): {predicted_distance}")
                    
                    if predicted_distance < self.clamp_distance_thres:
                        predicted_distance = None
                        print(f"[DEBUG dist()] Distance clamped to None (< {self.clamp_distance_thres})")

        print(f"[DEBUG dist()] Final distance: {predicted_distance}")
        return round(predicted_distance, 2) if predicted_distance is not None else None

    def _compute_dist_geometric_features(self, mask1, mask2, depth_map):
        """
        Compute geometric features for GeometryFusedResNet (Distance Model).
        
        Args:
            mask1: Binary mask [H, W] as numpy array
            mask2: Binary mask [H, W] as numpy array
            depth_map: Depth map [H, W] as numpy array, normalized to [0, 1]
        
        Returns:
            geo_features: Tensor of shape [14]
        """
        H, W = mask1.shape
        diag = np.sqrt(H**2 + W**2)
        
        mask1_bin = (mask1 > 0.5).astype(np.float32)
        mask2_bin = (mask2 > 0.5).astype(np.float32)
        
        features = []
        
        # Centroid distance
        coords1 = np.where(mask1_bin > 0)
        coords2 = np.where(mask2_bin > 0)
        
        if len(coords1[0]) > 0:
            centroid1 = np.array([np.mean(coords1[1]), np.mean(coords1[0])])
        else:
            centroid1 = np.array([W/2, H/2])
        
        if len(coords2[0]) > 0:
            centroid2 = np.array([np.mean(coords2[1]), np.mean(coords2[0])])
        else:
            centroid2 = np.array([W/2, H/2])
        
        centroid_dist = np.linalg.norm(centroid1 - centroid2) / diag
        features.append(centroid_dist)
        
        # Depth statistics
        if len(coords1[0]) > 0:
            mean_depth_1 = np.mean(depth_map[coords1[0], coords1[1]])
        else:
            mean_depth_1 = 0.0
        
        if len(coords2[0]) > 0:
            mean_depth_2 = np.mean(depth_map[coords2[0], coords2[1]])
        else:
            mean_depth_2 = 0.0
        
        depth_diff = abs(mean_depth_1 - mean_depth_2)
        features.extend([depth_diff, mean_depth_1, mean_depth_2])
        
        # Normalized areas
        area1 = np.sum(mask1_bin)
        area2 = np.sum(mask2_bin)
        features.extend([np.sqrt(area1) / diag, np.sqrt(area2) / diag])
        
        # Bounding boxes
        if len(coords1[0]) > 0:
            bbox1 = [np.min(coords1[1]) / W, np.min(coords1[0]) / H,
                    np.max(coords1[1]) / W, np.max(coords1[0]) / H]
        else:
            bbox1 = [0.0, 0.0, 0.0, 0.0]
        features.extend(bbox1)
        
        if len(coords2[0]) > 0:
            bbox2 = [np.min(coords2[1]) / W, np.min(coords2[0]) / H,
                    np.max(coords2[1]) / W, np.max(coords2[0]) / H]
        else:
            bbox2 = [0.0, 0.0, 0.0, 0.0]
        features.extend(bbox2)
        
        return torch.tensor(features, dtype=torch.float32)

    def _compute_simple_dist_geometric_features(self, mask1, mask2, depth_map):
        """
        Compute simple geometric features for ResNetWithShortcut (Distance Model).
        Returns only 3 features: [mean_depth_1, mean_depth_2, centroid_dist_2d]
        
        Args:
            mask1: Binary mask [H, W] as numpy array
            mask2: Binary mask [H, W] as numpy array
            depth_map: Depth map [H, W] as numpy array, normalized to [0, 1]
        
        Returns:
            geo_features: Tensor of shape [3]
        """
        H, W = mask1.shape
        diag = np.sqrt(H**2 + W**2)
        
        mask1_bin = (mask1 > 0.5).astype(np.float32)
        mask2_bin = (mask2 > 0.5).astype(np.float32)
        
        features = []
        
        # Feature 1: Mean depth of mask1
        coords1 = np.where(mask1_bin > 0)
        if len(coords1[0]) > 0:
            mean_depth_1 = np.mean(depth_map[coords1[0], coords1[1]])
        else:
            mean_depth_1 = 0.0
        features.append(mean_depth_1)
        
        # Feature 2: Mean depth of mask2
        coords2 = np.where(mask2_bin > 0)
        if len(coords2[0]) > 0:
            mean_depth_2 = np.mean(depth_map[coords2[0], coords2[1]])
        else:
            mean_depth_2 = 0.0
        features.append(mean_depth_2)
        
        # Feature 3: Centroid distance (2D)
        if len(coords1[0]) > 0:
            centroid1 = np.array([np.mean(coords1[1]), np.mean(coords1[0])])
        else:
            centroid1 = np.array([W/2, H/2])
        
        if len(coords2[0]) > 0:
            centroid2 = np.array([np.mean(coords2[1]), np.mean(coords2[0])])
        else:
            centroid2 = np.array([W/2, H/2])
        
        centroid_dist_2d = np.linalg.norm(centroid1 - centroid2) / diag
        features.append(centroid_dist_2d)
        
        return torch.tensor(features, dtype=torch.float32)

    def _centroid(self, mask_array: np.ndarray):
        """Compute the centroid (x, y) of a mask."""
        y_indices, x_indices = np.where(mask_array > 0)
        if len(x_indices) == 0:
            return (0, 0)
        return (np.mean(x_indices), np.mean(y_indices))

    def closest(self, mask_A: Mask, masks: List[Mask]) -> str:
        rgb = Image.open(self.img_path).convert('RGB')
        rgb = F.resize(rgb, self.resize)
        rgb = np.array(rgb).astype(np.float32) / 255.0

        # Load depth if using geometry (use closest model's geometry setting)
        depth_np = None
        if self.closest_use_geometry and self.depth_path and os.path.exists(self.depth_path):
            depth = Image.open(self.depth_path)
            depth = F.resize(depth, self.resize, interpolation=Image.BILINEAR)
            depth_np = np.array(depth).astype(np.float32) / 65535.0

        # Decode and resize mask_A
        maskA_array = mask_A.decode_mask()
        maskA_img = Image.fromarray(maskA_array)
        maskA_img = F.resize(maskA_img, self.resize, interpolation=Image.NEAREST)
        maskA_resized = np.array(maskA_img).astype(np.float32)

        # Prepare the batch
        batch_tensors = []
        batch_geo_features = []
        
        for m in masks:
            maskB_array = m.decode_mask()
            maskB_img = Image.fromarray(maskB_array)
            maskB_img = F.resize(maskB_img, self.resize, interpolation=Image.NEAREST)
            maskB_resized = np.array(maskB_img).astype(np.float32)

            # Stack inputs (use closest model's geometry setting)
            if self.closest_use_geometry and depth_np is not None:
                components = [rgb, depth_np[..., None], maskA_resized[..., None], maskB_resized[..., None]]
                # Compute geometric features based on model type
                if self.use_dedicated_closest:
                    # Dedicated closest model uses 7 geo features
                    geo_feat = self._compute_closest_geometric_features(maskA_resized, maskB_resized, depth_np)
                elif self.closest_use_shortcut:
                    geo_feat = self._compute_simple_dist_geometric_features(maskA_resized, maskB_resized, depth_np)
                else:
                    geo_feat = self._compute_dist_geometric_features(maskA_resized, maskB_resized, depth_np)
                batch_geo_features.append(geo_feat)
            else:
                components = [rgb, maskA_resized[..., None], maskB_resized[..., None]]
            
            input_tensor = np.concatenate(components, axis=-1)  # H x W x C
            input_tensor = torch.tensor(input_tensor).permute(2, 0, 1)  # C x H x W
            batch_tensors.append(input_tensor)

        batch_tensor = torch.stack(batch_tensors).to(DEVICE)  # N x C x H x W

        # Model inference (use closest model)
        with torch.no_grad():
            if self.closest_use_geometry and depth_np is not None:
                batch_geo_features = torch.stack(batch_geo_features).to(DEVICE)
                predicted_scores = self.closest_model(batch_tensor, batch_geo_features).cpu().numpy()
            else:
                predicted_scores = self.closest_model(batch_tensor).cpu().numpy()

        # Handle output based on model type
        if self.use_dedicated_closest:
            # Dedicated closest model outputs closeness score (0~1, higher=closer)
            # Find the mask with the HIGHEST score
            print(f"\n[DEBUG closest()] Predicted closeness scores for {mask_A.mask_name()}:")
            for i, (m, score) in enumerate(zip(masks, predicted_scores)):
                print(f"  {i}: {m.mask_name()} -> {score:.4f}")
            
            max_index = np.argmax(predicted_scores)
            closest_mask = masks[max_index]
            print(f"  => Closest: {closest_mask.mask_name()} (score: {predicted_scores[max_index]:.4f})\n")
        else:
            # Distance model outputs distance (lower=closer)
            # 只有 baseline model (沒用 geometry) 才需要除以 100
            if not self.closest_use_geometry:
                predicted_scores = predicted_scores / 100.0
            
            # Clamp negative distances to 0 (distance should be non-negative)
            predicted_scores = np.maximum(predicted_scores, 0.0)
            
            print(f"\n[DEBUG closest()] Predicted distances for {mask_A.mask_name()}:")
            for i, (m, dist) in enumerate(zip(masks, predicted_scores)):
                print(f"  {i}: {m.mask_name()} -> {dist:.2f}cm")
            
            min_index = np.argmin(predicted_scores)
            closest_mask = masks[min_index]
            print(f"  => Closest: {closest_mask.mask_name()} (distance: {predicted_scores[min_index]:.2f}cm)\n")

        return closest_mask.mask_name()
    
    def _compute_closest_geometric_features(self, mask1, mask2, depth_map):
        """
        Compute geometric features for dedicated closest model (7 features).
        
        Returns:
            geo_features: Tensor of shape [7]
        """
        H, W = mask1.shape
        diag = np.sqrt(H**2 + W**2)
        
        mask1_bin = (mask1 > 0.5).astype(np.float32)
        mask2_bin = (mask2 > 0.5).astype(np.float32)
        
        features = []
        coords1 = np.where(mask1_bin > 0)
        coords2 = np.where(mask2_bin > 0)
        
        if len(coords1[0]) > 0:
            mean_depth_1 = np.mean(depth_map[coords1[0], coords1[1]])
            centroid1 = np.array([np.mean(coords1[1]), np.mean(coords1[0])])
            area1 = len(coords1[0])
        else:
            mean_depth_1 = 0.5
            centroid1 = np.array([W/2, H/2])
            area1 = 1
        
        if len(coords2[0]) > 0:
            mean_depth_2 = np.mean(depth_map[coords2[0], coords2[1]])
            centroid2 = np.array([np.mean(coords2[1]), np.mean(coords2[0])])
            area2 = len(coords2[0])
        else:
            mean_depth_2 = 0.5
            centroid2 = np.array([W/2, H/2])
            area2 = 1
        
        features.append(mean_depth_1)
        features.append(mean_depth_2)
        features.append(np.linalg.norm(centroid1 - centroid2) / diag)
        features.append(np.clip(np.sqrt(area1) / np.sqrt(area2) if area2 > 0 else 1.0, 0.1, 10.0))
        features.append(abs(mean_depth_1 - mean_depth_2))
        features.append((centroid1[0] - centroid2[0]) / W)
        features.append((centroid1[1] - centroid2[1]) / H)
        
        return torch.tensor(features, dtype=torch.float32)


    def is_left(self, mask_A: Mask, mask_B) -> bool:
        """檢查 mask_A 是否在 mask_B 的左側
        
        Args:
            mask_A: 要檢查的 mask
            mask_B: 參考 mask，可以是單個 Mask 或 Mask 列表
        
        Returns:
            如果 mask_B 是列表，返回 mask_A 是否在所有 mask_B 的左側
            如果 mask_B 是單個 mask，返回 mask_A 是否在 mask_B 的左側
        """
        centroid_A = self._centroid(mask_A.decode_mask())
        
        if isinstance(mask_B, list):
            # 如果是列表，檢查是否在所有 mask 的左側
            for mask in mask_B:
                centroid_B = self._centroid(mask.decode_mask())
                if centroid_A[0] >= centroid_B[0]:  # 不在這個 mask 的左側
                    return False
            return True  # 在所有 mask 的左側
        else:
            # 單個 mask 的情況
            centroid_B = self._centroid(mask_B.decode_mask())
            return centroid_A[0] < centroid_B[0]

    def is_right(self, mask_A: Mask, mask_B) -> bool:
        """檢查 mask_A 是否在 mask_B 的右側
        
        Args:
            mask_A: 要檢查的 mask
            mask_B: 參考 mask，可以是單個 Mask 或 Mask 列表
        
        Returns:
            如果 mask_B 是列表，返回 mask_A 是否在所有 mask_B 的右側
            如果 mask_B 是單個 mask，返回 mask_A 是否在 mask_B 的右側
        """
        centroid_A = self._centroid(mask_A.decode_mask())
        
        if isinstance(mask_B, list):
            # 如果是列表，檢查是否在所有 mask 的右側
            for mask in mask_B:
                centroid_B = self._centroid(mask.decode_mask())
                if centroid_A[0] <= centroid_B[0]:  # 不在這個 mask 的右側
                    return False
            return True  # 在所有 mask 的右側
        else:
            # 單個 mask 的情況
            centroid_B = self._centroid(mask_B.decode_mask())
            return centroid_A[0] > centroid_B[0]
    
    def mask_IoU(self, mask_A: Mask, mask_B: Mask) -> float:
        maskA_array = mask_A.decode_mask()
        maskB_array = mask_B.decode_mask()
        intersection = np.logical_and(maskA_array, maskB_array).sum()
        union = np.logical_or(maskA_array, maskB_array).sum()
        iou = intersection / union if union > 0 else 0.0
        return iou

    # def inside(self, mask_A: Mask, masks: List[Mask], debug=False) -> int:
    #     rgb = Image.open(self.img_path).convert('RGB')
    #     rgb = F.resize(rgb, self.resize)
    #     rgb = np.array(rgb).astype(np.float32) / 255.0

    #     # Decode and resize mask_A
    #     maskA_array = mask_A.decode_mask()
    #     maskA_img = Image.fromarray(maskA_array)
    #     maskA_img = F.resize(maskA_img, self.resize, interpolation=Image.NEAREST)
    #     maskA_resized = np.array(maskA_img).astype(np.float32)

    #     batch_tensors = []
    #     mask_names = []

    #     for m in masks:
    #         maskB_array = m.decode_mask()
    #         maskB_img = Image.fromarray(maskB_array)
    #         maskB_img = F.resize(maskB_img, self.resize, interpolation=Image.NEAREST)
    #         maskB_resized = np.array(maskB_img).astype(np.float32)

    #         # Stack RGB, maskA, maskB as channels (adapt as needed)
    #         components = [rgb, maskA_resized[..., None], maskB_resized[..., None]]
    #         input_tensor = np.concatenate(components, axis=-1)
    #         input_tensor = torch.tensor(input_tensor).permute(2, 0, 1)
    #         batch_tensors.append(input_tensor)
    #         mask_names.append(m.mask_name())

    #     batch_tensor = torch.stack(batch_tensors).to(DEVICE)

    #     with torch.no_grad():
    #         # Output: logits, convert to 0/1 using configurable threshold
    #         logits = self.inside_model(batch_tensor)
    #         preds = torch.sigmoid(logits).cpu().numpy()
            
    #         # Debug: print probability scores
    #         if debug:
    #             print(f"\n=== Inside Debug for {mask_A.mask_name()} ===")
    #             for i, (name, prob) in enumerate(zip(mask_names, preds)):
    #                 inside_label = "INSIDE" if prob >= self.inside_thres else "OUTSIDE"
    #                 print(f"  {name}: prob={prob:.4f} ({inside_label})")
    #             print(f"  Threshold: {self.inside_thres}")
            
    #         # Apply threshold
    #         preds_binary = (preds >= self.inside_thres).astype(np.int32)

    #     count = int(preds_binary.sum())
        
    #     if debug:
    #         print(f"  Total count: {count}/{len(masks)}")
        
    #     return count
    def inside(self, mask_A: Mask, masks: List[Mask], debug=False) -> int:
        rgb = Image.open(self.img_path).convert('RGB')
        rgb = F.resize(rgb, self.resize)
        rgb = np.array(rgb).astype(np.float32) / 255.0

        # Decode and resize mask_A
        maskA_array = mask_A.decode_mask()
        maskA_img = Image.fromarray(maskA_array)
        maskA_img = F.resize(maskA_img, self.resize, interpolation=Image.NEAREST)
        maskA_resized = np.array(maskA_img).astype(np.float32)

        # Optional: Load depth if using geometry
        depth = None
        if self.use_geometry and self.depth_path is not None and os.path.exists(self.depth_path):
            depth = Image.open(self.depth_path)
            depth = F.resize(depth, self.resize)
            depth = np.array(depth).astype(np.float32)

        batch_tensors = []
        batch_geo_features = []
        mask_names = []

        for m in masks:
            maskB_array = m.decode_mask()
            maskB_img = Image.fromarray(maskB_array)
            maskB_img = F.resize(maskB_img, self.resize, interpolation=Image.NEAREST)
            maskB_resized = np.array(maskB_img).astype(np.float32)

            # Stack RGB, maskA, maskB as channels
            components = [rgb, maskA_resized[..., None], maskB_resized[..., None]]
            input_tensor = np.concatenate(components, axis=-1)
            input_tensor = torch.tensor(input_tensor).permute(2, 0, 1)
            batch_tensors.append(input_tensor)
            mask_names.append(m.mask_name())
            
            # Compute geometric features if using dual-stream
            if self.use_geometry:
                geo_feat = self._compute_inside_geometric_features(
                    maskA_resized, maskB_resized, depth
                )
                batch_geo_features.append(geo_feat)

        batch_tensor = torch.stack(batch_tensors).to(DEVICE)

        with torch.no_grad():
            if self.use_geometry:
                batch_geo_tensor = torch.stack(batch_geo_features).to(DEVICE)
                logits = self.inside_model(batch_tensor, batch_geo_tensor)
            else:
                logits = self.inside_model(batch_tensor)
            
            preds = torch.sigmoid(logits).cpu().numpy()
            
            # Debug: print probability scores
            if debug:
                print(f"\n=== Inside Debug for {mask_A.mask_name()} ===")
                for i, (name, prob) in enumerate(zip(mask_names, preds)):
                    inside_label = "INSIDE" if prob >= self.inside_thres else "OUTSIDE"
                    print(f"  {name}: prob={prob:.4f} ({inside_label})")
                print(f"  Threshold: {self.inside_thres}")
            
            # Apply threshold with additional IoU minimum check
            # Require at least 10% of candidate mask to overlap with reference mask
            preds_binary = np.zeros(len(preds), dtype=np.int32)
            for idx, (prob, m) in enumerate(zip(preds, masks)):
                if prob >= self.inside_thres:
                    # Additional IoU check: compute overlap ratio
                    maskB_array = m.decode_mask()
                    maskA_bin = maskA_array > 0
                    maskB_bin = maskB_array > 0
                    intersection = np.logical_and(maskA_bin, maskB_bin).sum()
                    maskB_area = maskB_bin.sum()
                    overlap_ratio = intersection / maskB_area if maskB_area > 0 else 0
                    
                    # Require at least 5% overlap to count as inside (very lenient to reduce under-counting)
                    if overlap_ratio >= 0.05:
                        preds_binary[idx] = 1
                        if debug:
                            print(f"  {mask_names[idx]}: overlap_ratio={overlap_ratio:.3f} -> INSIDE")
                    else:
                        if debug:
                            print(f"  {mask_names[idx]}: overlap_ratio={overlap_ratio:.3f} -> REJECTED (< 0.05)")

        count = int(preds_binary.sum())
        
        if debug:
            print(f"  Total count: {count}/{len(masks)}")
        
        return count

    def _compute_inside_geometric_features(self, obj_mask, buffer_mask, depth=None):
        """
        Compute 8 geometric features for dual-stream inside prediction model.
        
        Args:
            obj_mask: [H, W] numpy array (normalized 0-1)
            buffer_mask: [H, W] numpy array (normalized 0-1)
            depth: [H, W] numpy array (optional, for depth difference)
        
        Returns:
            torch.Tensor of shape [8]
        """
        H, W = obj_mask.shape
        img_area = H * W
        
        # Convert to binary masks
        obj_binary = (obj_mask > 0.5).astype(np.float32)
        buffer_binary = (buffer_mask > 0.5).astype(np.float32)
        
        # Feature 1: IoU
        intersection = np.sum(obj_binary * buffer_binary)
        union = np.sum(np.maximum(obj_binary, buffer_binary))
        iou = intersection / union if union > 0 else 0.0
        
        # Feature 2-3: Area ratios
        obj_area = np.sum(obj_binary)
        buffer_area = np.sum(buffer_binary)
        obj_area_norm = obj_area / img_area
        buffer_area_norm = buffer_area / img_area
        
        # Feature 4-5: Overlap ratios
        overlap_obj_ratio = intersection / obj_area if obj_area > 0 else 0.0
        overlap_buffer_ratio = intersection / buffer_area if buffer_area > 0 else 0.0
        
        # Feature 6-7: Object center (normalized)
        y_coords, x_coords = np.where(obj_binary > 0)
        if len(x_coords) > 0:
            center_x = np.mean(x_coords) / W
            center_y = np.mean(y_coords) / H
        else:
            center_x, center_y = 0.5, 0.5
        
        # Feature 8: Depth difference
        if depth is not None:
            obj_depth_mean = np.mean(depth[obj_binary > 0]) if np.sum(obj_binary) > 0 else 0
            buffer_depth_mean = np.mean(depth[buffer_binary > 0]) if np.sum(buffer_binary) > 0 else 0
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
    def most_right(self, masks: List[Mask]) -> int:
        max_x = -1
        rightmost_id = -1
        rightmost_mask = None
        for m in masks:
            centroid = self._centroid(m.decode_mask())
            if centroid[0] > max_x:
                max_x = centroid[0]
                rightmost_id = m.object_id
                rightmost_mask = m
        return rightmost_mask.mask_name()

    def most_left(self, masks: List[Mask]) -> int:
        min_x = float('inf')
        leftmost_id = -1
        leftmost_mask = None
        for m in masks:
            centroid = self._centroid(m.decode_mask())
            if centroid[0] < min_x:
                min_x = centroid[0]
                leftmost_id = m.object_id
                leftmost_mask = m
        return leftmost_mask.mask_name()
    
    def middle(self, masks: List[Mask]) -> str:
        # return the mask that is in the middle of all masks
        if not masks:
            raise ValueError("No masks provided to find the middle mask.")
        
        if len(masks) == 1:
            # 只有一個 mask，直接返回
            return masks[0].mask_name()
        
        centroids = []
        for m in masks:
            centroid_x, _ = self._centroid(m.decode_mask())
            centroids.append((centroid_x, m))

        # Sort masks by centroid_x
        centroids.sort(key=lambda x: x[0])

        # Return the mask name of the middle one
        # 如果奇數個，返回中間的；如果偶數個，返回中間偏右的（索引 len//2）
        middle_index = len(centroids) // 2
        middle_mask = centroids[middle_index][1]
        return middle_mask.mask_name()

    def is_empty(self, transporter_masks: List[Mask]) -> str:
        """
        For each transporter mask, calculate the maximum IoU with all pallet masks.
        Return the transporter mask with the smallest maximum IoU.
        """
        min_max_IoU = float('inf')
        selected_transporter = None

        pallet_masks = [m for m in self.masks.values() if 'pallet' in m.object_class.lower()]

        for transporter in transporter_masks:
            if transporter.object_class.lower() != 'transporter':
                continue

            max_IoU = 0.0
            for pallet in pallet_masks:
                iou = self.mask_IoU(transporter, pallet)
                if iou > max_IoU:
                    max_IoU = iou

            if max_IoU < min_max_IoU:
                min_max_IoU = max_IoU
                selected_transporter = transporter

        if selected_transporter is not None:
            return selected_transporter.mask_name()
        else:
            raise ValueError("No transporter masks found in the provided list.")