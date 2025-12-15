"""
比較有加depth訓練的distance estimation model和原版的inference準確率
- 有加depth的model: /tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/agent/ckpt_ours/best_model.pth (GeometryFusedResNet)
  Input: RGB + Depth + 2 Masks (6 channels) + geometric features
- 原版的model: /tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/distance_est/ckpt/epoch_5_iter_6831.pth (ResNetDistanceRegressor)
  Input: RGB + 2 Masks (5 channels), NO depth
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ResNetDistanceRegressor, GeometryFusedResNet
import numpy as np
from tqdm import tqdm
import json
import os
from PIL import Image
import torchvision.transforms.functional as F
import pycocotools.mask as mask_utils
from data_loader import DistanceDataset

def evaluate_model(model, loader, device, use_geometry=False, model_name="Model"):
    """評估單個模型的準確率"""
    model.eval()
    
    all_preds = []
    all_targets = []
    errors = []
    
    print(f"\n正在評估 {model_name}...")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference {model_name}"):
            if use_geometry:
                inputs, geo_features, targets = batch
                inputs = inputs.to(device, non_blocking=True)
                geo_features = geo_features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds = model(inputs, geo_features)
            else:
                inputs, targets = batch
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds = model(inputs) / 100
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Calculate errors
            error = torch.abs(preds - targets)
            errors.extend(error.cpu().numpy())
    
    if len(all_preds) == 0:
        print(f"[ERROR] No predictions collected for {model_name}!")
        return None, None, None, None
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    errors = np.array(errors)
    
    # 計算各種metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    median_error = np.median(errors)
    
    # 計算在不同threshold下的準確率
    thresholds = [0.5, 1.0, 2.0, 5.0]
    accuracies = {}
    for thresh in thresholds:
        acc = np.mean(errors < thresh) * 100
        accuracies[f"acc@{thresh}m"] = acc
    
    results = {
        "model_name": model_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "median_error": float(median_error),
        "accuracies": accuracies,
        "num_samples": len(all_targets)
    }
    
    return results, all_preds, all_targets, errors


def print_results(results):
    """打印結果"""
    print(f"\n{'='*70}")
    print(f"Model: {results['model_name']}")
    print(f"{'='*70}")

    print(f"Number of samples: {results['num_samples']}")
    print(f"\nError Metrics:")
    print(f"  Mean Absolute Error (MAE):     {results['mae']:.4f} m")
    print(f"  Root Mean Square Error (RMSE): {results['rmse']:.4f} m")
    print(f"  Median Error:                  {results['median_error']:.4f} m")
    print(f"\nAccuracy at different thresholds:")
    for key, val in results['accuracies'].items():
        print(f"  {key}: {val:.2f}%")


def compare_models(results_depth, results_baseline):
    """比較兩個模型的結果"""
    print(f"\n{'='*70}")
    print(f"模型比較")
    print(f"{'='*70}")
    
    print(f"\n[With Depth] vs [Baseline]:")
    print(f"\nMAE: {results_depth['mae']:.4f} vs {results_baseline['mae']:.4f}")
    mae_improvement = ((results_baseline['mae'] - results_depth['mae']) / results_baseline['mae']) * 100
    print(f"  → Improvement: {mae_improvement:+.2f}%")
    
    print(f"\nRMSE: {results_depth['rmse']:.4f} vs {results_baseline['rmse']:.4f}")
    rmse_improvement = ((results_baseline['rmse'] - results_depth['rmse']) / results_baseline['rmse']) * 100
    print(f"  → Improvement: {rmse_improvement:+.2f}%")
    
    print(f"\nMedian Error: {results_depth['median_error']:.4f} vs {results_baseline['median_error']:.4f}")
    median_improvement = ((results_baseline['median_error'] - results_depth['median_error']) / results_baseline['median_error']) * 100
    print(f"  → Improvement: {median_improvement:+.2f}%")
    
    print("\nAccuracy comparison:")
    for key in results_depth['accuracies'].keys():
        acc_depth = results_depth['accuracies'][key]
        acc_baseline = results_baseline['accuracies'][key]
        diff = acc_depth - acc_baseline
        print(f"  {key}: {acc_depth:.2f}% vs {acc_baseline:.2f}% (diff: {diff:+.2f}%)")


def main():
    # ===== Configuration =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    data_dir = "/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/data/val"
    val_json = "/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/distance_est/val_dist_est.json"
    
    ckpt_with_depth = "/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/agent/ckpt_ours/best_model.pth"
    ckpt_baseline = "/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/distance_est/ckpt/epoch_5_iter_6831.pth"
    
    # ===== Load Dataset =====
    print("\n載入驗證數據集...")
    
    # Dataset for model with geometry (6-channel input + geo features)
    val_dataset_geo = DistanceDataset(
        data_dir=data_dir,
        json_path=val_json,
        rgb=True,
        depth=True,
        use_geometry=True,  # Enable geometry features
        max_depth=65535.0,
        distance_scale=1.0
    )
    
    # Dataset for baseline model (5-channel input: RGB + 2 Masks, no depth)
    val_dataset_baseline = DistanceDataset(
        data_dir=data_dir,
        json_path=val_json,
        rgb=True,
        depth=False,  # Baseline doesn't use depth
        use_geometry=False,
        max_depth=65535.0,
        distance_scale=1.0
    )
    
    # Create dataloaders
    batch_size = 16
    val_loader_geo = DataLoader(
        val_dataset_geo,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader_baseline = DataLoader(
        val_dataset_baseline,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # ===== Load Model with Depth (GeometryFusedResNet) =====
    print("\n載入有加depth的模型 (GeometryFusedResNet)...")
    model_with_depth = GeometryFusedResNet(
        input_channels=6,  # RGB + Depth + 2 Masks
        num_geo_features=14,
        backbone='resnet50',
        pretrained=False
    )
    
    checkpoint = torch.load(ckpt_with_depth, map_location=device)
    model_with_depth.load_state_dict(checkpoint)
    model_with_depth.to(device)
    model_with_depth.eval()
    print(f"✓ 模型載入成功: {ckpt_with_depth}")
    
    # ===== Load Baseline Model (ResNetDistanceRegressor) =====
    print("\n載入原版模型 (ResNetDistanceRegressor)...")
    model_baseline = ResNetDistanceRegressor(
        input_channels=5,  # RGB + 2 Masks (no depth)
        backbone='resnet50',
        pretrained=False
    )
    
    checkpoint = torch.load(ckpt_baseline, map_location=device)
    model_baseline.load_state_dict(checkpoint)
    model_baseline.to(device)
    model_baseline.eval()
    print(f"✓ 模型載入成功: {ckpt_baseline}")
    
    # ===== Evaluate Models =====
    results_with_depth, preds1, targets1, errors1 = evaluate_model(
        model_with_depth, 
        val_loader_geo, 
        device, 
        use_geometry=True,
        model_name="Model with Depth (GeometryFusedResNet)"
    )
    
    results_baseline, preds2, targets2, errors2 = evaluate_model(
        model_baseline,
        val_loader_baseline,
        device,
        use_geometry=False,
        model_name="Baseline Model (ResNetDistanceRegressor)"
    )
    
    # ===== Print Results =====
    print_results(results_with_depth)
    print_results(results_baseline)
    compare_models(results_with_depth, results_baseline)
    
    # ===== Save Results to JSON =====
    output_file = "/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/distance_est/comparison_results.json"
    comparison_data = {
        "model_with_depth": results_with_depth,
        "baseline_model": results_baseline,
        "comparison": {
            "mae_improvement_%": float(((results_baseline['mae'] - results_with_depth['mae']) / results_baseline['mae']) * 100),
            "rmse_improvement_%": float(((results_baseline['rmse'] - results_with_depth['rmse']) / results_baseline['rmse']) * 100),
            "median_improvement_%": float(((results_baseline['median_error'] - results_with_depth['median_error']) / results_baseline['median_error']) * 100)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✓ 結果已保存至: {output_file}")


if __name__ == "__main__":
    main()
