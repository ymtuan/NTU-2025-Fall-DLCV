import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from data_loader import DistanceDataset
from model import ResNetDistanceRegressor, GeometryFusedResNet, ResNetWithShortcut
import argparse
import os
import numpy as np
from tqdm import tqdm


class LogSpaceMSELoss(nn.Module):
    """
    Log-Space MSE Loss for better sensitivity to small distance errors.
    
    Standard MSE is sensitive to large values but less sensitive to small values.
    This loss function transforms both predictions and targets to log space,
    making the model more sensitive to percentage errors in small distances.
    
    Formula: MSE(log(1 + pred), log(1 + target))
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Transform to log space using log1p to avoid log(0)
        # log1p(x) = log(1 + x)
        log_pred = torch.log1p(pred) 
        log_target = torch.log1p(target)
        return self.mse(log_pred, log_target)

class WeightedLogSpaceMSELoss(nn.Module):
    def __init__(self, threshold=1.5, weight=3.0, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none') # 記得改成 none 以便逐樣本加權
        self.threshold = threshold
        self.weight = weight
        self.eps = eps

    def forward(self, pred, target):
        # 確保 pred 和 target 都是非負的（避免 log1p 產生 NaN）
        # 使用 clamp 確保最小值為 0（log1p(0) = 0，不會有問題）
        pred_clamped = torch.clamp(pred, min=0.0)
        target_clamped = torch.clamp(target, min=0.0)
        
        # 檢查是否有 NaN 或 Inf
        if torch.isnan(pred_clamped).any() or torch.isinf(pred_clamped).any():
            print(f"[WARNING] pred contains NaN/Inf: min={pred.min().item():.4f}, max={pred.max().item():.4f}")
            pred_clamped = torch.clamp(pred_clamped, min=0.0, max=1e6)
        
        if torch.isnan(target_clamped).any() or torch.isinf(target_clamped).any():
            print(f"[WARNING] target contains NaN/Inf: min={target.min().item():.4f}, max={target.max().item():.4f}")
            target_clamped = torch.clamp(target_clamped, min=0.0, max=1e6)
        
        # 轉換到 log space
        log_pred = torch.log1p(pred_clamped + self.eps) 
        log_target = torch.log1p(target_clamped + self.eps)
        
        # 計算 MSE loss
        loss = self.mse(log_pred, log_target)
        
        # 檢查 loss 是否有 NaN
        if torch.isnan(loss).any():
            print(f"[ERROR] Loss contains NaN after log1p!")
            print(f"  pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
            print(f"  target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
            print(f"  pred_clamped range: [{pred_clamped.min().item():.4f}, {pred_clamped.max().item():.4f}]")
            print(f"  target_clamped range: [{target_clamped.min().item():.4f}, {target_clamped.max().item():.4f}]")
            # 將 NaN 替換為 0
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)
        
        # 建立權重向量：如果 target < threshold，權重設為 weight，否則 1.0
        weights = torch.ones_like(loss)
        weights[target_clamped < self.threshold] = self.weight
        
        # 回傳加權後的平均 Loss
        final_loss = (loss * weights).mean()
        
        # 最終檢查
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            print(f"[ERROR] Final loss is NaN/Inf! Returning fallback loss.")
            return torch.tensor(1e6, device=final_loss.device, requires_grad=True)
        
        return final_loss

def train_epoch(model, loader, optimizer, criterion, device, use_geometry=False, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(loader)
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        if use_geometry:
            inputs, geo_features, targets = batch
            inputs = inputs.to(device, non_blocking=True)
            geo_features = geo_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        else:
            inputs, targets = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=(scaler is not None)):
            if use_geometry:
                preds = model(inputs, geo_features)
            else:
                preds = model(inputs)
            loss = criterion(preds, targets)
        
        # 檢查 loss 是否為 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[ERROR] NaN/Inf loss detected! Skipping batch.")
            print(f"  Loss value: {loss.item()}")
            continue
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            # 梯度裁剪（防止梯度爆炸）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, loader, criterion, device, use_geometry=False):
    """Validation loop"""
    model.eval()
    total_loss = 0
    num_batches = len(loader)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
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
                preds = model(inputs)
            
            loss = criterion(preds, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ===== Dataset =====
    train_dataset = DistanceDataset(
        data_dir=args.data_dir,
        json_path=args.train_json,
        rgb=True,
        depth=True,
        use_geometry=args.use_geometry,
        simple_geo_features=args.use_shortcut if args.use_geometry else False,
        max_depth=args.max_depth,
        distance_scale=args.distance_scale
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if args.workers > 0 else False
    )
    
    # Validation dataset (optional)
    val_loader = None
    if args.val_json:
        val_dataset = DistanceDataset(
            data_dir=args.data_dir.replace('train', 'val'),
            json_path=args.val_json,
            rgb=True,
            depth=True,
            use_geometry=args.use_geometry,
            simple_geo_features=args.use_shortcut if args.use_geometry else False,
            max_depth=args.max_depth,
            distance_scale=args.distance_scale
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
    
    # ===== Model =====
    if args.use_geometry:
        if args.use_shortcut:
            print("\n✓ Using ResNetWithShortcut (Shortcut Architecture)")
            print(f"  - Input channels: 6 (RGB + Depth + 2 Masks)")
            print(f"  - Geometric features: 3 [mean_depth_1, mean_depth_2, centroid_dist_2d]")
            model = ResNetWithShortcut(
                in_channels=6,
                num_geo_features=3,
                backbone='resnet50',
                pretrained=args.pretrained
            )
        else:
            print("\n✓ Using GeometryFusedResNet (Dual-Stream Architecture)")
            print(f"  - Input channels: 6 (RGB + Depth + 2 Masks)")
            print(f"  - Geometric features: 14")
            model = GeometryFusedResNet(
                input_channels=6,
                num_geo_features=14,
                backbone='resnet50',
                pretrained=args.pretrained
            )
    else:
        print("\n✓ Using ResNetDistanceRegressor (Baseline)")
        model = ResNetDistanceRegressor(
            input_channels=5,
            backbone='resnet50',
            pretrained=args.pretrained
        )
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}\n")
    
    # ===== Optimizer & Loss =====
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Select loss function
    if args.loss_type == 'log_mse':
        criterion = LogSpaceMSELoss()
        print("✓ Using Log-Space MSE Loss (better for small distance errors)\n")
    elif args.loss_type == 'weighted_log_mse':
        criterion = WeightedLogSpaceMSELoss(threshold=args.loss_threshold, weight=args.loss_weight)
        print(f"✓ Using Weighted Log-Space MSE Loss")
        print(f"  - Threshold: {args.loss_threshold} (samples < threshold get weight {args.loss_weight}x)\n")
    elif args.loss_type == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
        print("✓ Using Smooth L1 Loss (robust to outliers)\n")
    else:  # 'mse'
        criterion = nn.MSELoss()
        print("✓ Using Standard MSE Loss\n")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Mixed precision training
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        print("✓ Mixed Precision Training (AMP) enabled\n")
    
    # ===== TensorBoard =====
    log_dir = os.path.join(args.save_dir, 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"✓ TensorBoard logging to: {log_dir}")
    print(f"  Run: tensorboard --logdir={log_dir}\n")
    
    # ===== Training Loop =====
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_geometry=args.use_geometry, scaler=scaler
        )
        
        # 檢查 train_loss 是否為 NaN
        if np.isnan(train_loss) or np.isinf(train_loss):
            print(f"[ERROR] Train loss is NaN/Inf: {train_loss}")
            print("Stopping training due to NaN loss.")
            break
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validate
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device, use_geometry=args.use_geometry)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Log to TensorBoard
            writer.add_scalar('Loss/val', val_loss, epoch)
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save(model.state_dict(), save_path)
                print(f"✓ Best model saved to {save_path}")
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save epoch checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✓ Checkpoint saved to {save_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    if val_loader:
        print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Distance Estimation Model")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="../../DLCV_Final1/train",
                        help="Path to training data directory")
    parser.add_argument("--train_json", type=str, default="train_distance_pairs.json",
                        help="Path to training JSON")
    parser.add_argument("--val_json", type=str, default="val_distance_pairs.json",
                        help="Path to validation JSON (optional)")
    parser.add_argument("--max_depth", type=float, default=65535.0,
                        help="Maximum depth value for normalization")
    parser.add_argument("--distance_scale", type=float, default=1.0,
                        help="Scale factor for distance labels")
    
    # Model
    parser.add_argument("--use_geometry", action="store_true",
                        help="Use geometry-aware model instead of baseline")
    parser.add_argument("--use_shortcut", action="store_true",
                        help="Use ResNetWithShortcut (simple shortcut) instead of GeometryFusedResNet (dual-stream)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained weights")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--loss_type", type=str, default="mse",
                        choices=["mse", "log_mse", "weighted_log_mse", "smooth_l1"],
                        help="Loss function: mse (standard), log_mse (better for small distances), weighted_log_mse (weighted log-space MSE), smooth_l1 (robust to outliers)")
    parser.add_argument("--loss_threshold", type=float, default=2.5,
                        help="Threshold for weighted_log_mse loss (samples with target < threshold get higher weight)")
    parser.add_argument("--loss_weight", type=float, default=4.0,
                        help="Weight multiplier for weighted_log_mse loss (applied to samples below threshold)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision training")
    
    # Saving
    parser.add_argument("--save_dir", type=str, default="ckpt_log_mse",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=2,
                        help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    main(args)