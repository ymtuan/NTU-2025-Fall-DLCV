import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from data_loader import DistanceDataset
from model import ResNetDistanceRegressor, GeometryFusedResNet
import argparse
import os
from tqdm import tqdm


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
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
    criterion = nn.MSELoss()
    
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
    parser.add_argument("--data_dir", type=str, default="/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/data/train",
                        help="Path to training data directory")
    parser.add_argument("--train_json", type=str, default="/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/distance_est/train_dist_est.json",
                        help="Path to training JSON")
    parser.add_argument("--val_json", type=str, default="/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/distance_est/val_dist_est.json",
                        help="Path to validation JSON (optional)")
    parser.add_argument("--max_depth", type=float, default=65535.0,
                        help="Maximum depth value for normalization")
    parser.add_argument("--distance_scale", type=float, default=1.0,
                        help="Scale factor for distance labels")
    
    # Model
    parser.add_argument("--use_geometry", action="store_true",
                        help="Use GeometryFusedResNet (dual-stream) instead of baseline")
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
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision training")
    
    # Saving
    parser.add_argument("--save_dir", type=str, default="ckpt_ours",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=2,
                        help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    main(args)