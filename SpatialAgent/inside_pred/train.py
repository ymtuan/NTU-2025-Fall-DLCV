import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from data_loader import InsideDataset
from model import ResNet50Binary, GeometryAwareInclusionModel
from loss import FocalLoss
import os

def evaluate(model, loader, criterion, device, use_geometry=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            if use_geometry:
                inputs, geo_features, labels = batch
                inputs = inputs.to(device, non_blocking=True)
                geo_features = geo_features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs, geo_features)
            else:
                inputs, labels = batch
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    loss = running_loss / total
    acc = correct / total
    return loss, acc

def train_one_epoch(model, loader, optimizer, criterion, device, use_geometry=False, scaler=None, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    optimizer.zero_grad()  # Move outside loop for gradient accumulation
    
    for batch_idx, batch in enumerate(pbar):
        if use_geometry:
            inputs, geo_features, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            geo_features = geo_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
        # Mixed precision training
        with autocast(enabled=(scaler is not None)):
            if use_geometry:
                outputs = model(inputs, geo_features)
            else:
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Normalize loss for gradient accumulation
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation: only update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * inputs.size(0) * accumulation_steps

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = 100 * correct / total if total > 0 else 0
        pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{acc:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def get_val_path(path):
    return path.replace("train", "val")

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training dataset & loader
    dataset = InsideDataset(
        json_path=args.json,
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        use_depth=False,
        use_geometry=args.use_geometry,
        max_samples=args.max_samples
    )
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if args.workers > 0 else False
    )

    # Validation dataset & loader
    val_json = get_val_path(args.json)
    val_image_dir = get_val_path(args.image_dir)
    val_depth_dir = get_val_path(args.depth_dir)
    val_dataset = InsideDataset(
        json_path=val_json,
        image_dir=val_image_dir,
        depth_dir=val_depth_dir,
        use_depth=False,
        use_geometry=args.use_geometry,
        max_samples=args.max_val_samples
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2,  # Larger batch for validation (no gradient)
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if args.workers > 0 else False
    )

    # Model selection
    if args.use_geometry:
        print("Using GeometryAwareInclusionModel (Dual-Stream Architecture)")
        model = GeometryAwareInclusionModel(in_channels=5, num_geo_features=8)
    else:
        print("Using ResNet50Binary (Baseline)")
        model = ResNet50Binary(in_channels=5)
    
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        print("✓ Mixed Precision Training (AMP) enabled")

    os.makedirs(args.save_path, exist_ok=True)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Check if it's a full checkpoint or just model weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint format (new format)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load scaler state if using AMP
                if args.use_amp and 'scaler_state_dict' in checkpoint and scaler is not None:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                # Load training state
                start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                
                print(f"Resumed from epoch {start_epoch}, best_val_acc: {best_val_acc*100:.2f}%")
            else:
                # Old format: just model weights (state_dict directly)
                print("Detected old checkpoint format (model weights only). Loading model weights...")
                model.load_state_dict(checkpoint)
                print("Model weights loaded. Starting from epoch 0 (optimizer state not available).")
        else:
            print(f"Warning: Checkpoint {args.resume} not found. Starting from scratch.")
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            args.use_geometry, scaler, args.accumulation_steps
        )
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args.use_geometry)
        print(f"Epoch {epoch+1}/{args.epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Save checkpoint (full training state)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(args)  # Save arguments for reference
        }
        
        # Add scaler state if using AMP
        if args.use_amp and scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        # Save epoch checkpoint
        torch.save(checkpoint, f"{args.save_path}/epoch_{epoch+1}.pth")
        
        # Save latest checkpoint
        torch.save(checkpoint, f"{args.save_path}/latest.pth")
        
        # Save best model (state dict only for compatibility)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.save_path}/best_model.pth")
            # Also save full checkpoint for best model
            checkpoint['best_val_acc'] = best_val_acc
            torch.save(checkpoint, f"{args.save_path}/best_checkpoint.pth")
            print(f"✓ New best model saved! Val Acc: {val_acc*100:.2f}%")

    print(f"\nTraining completed. Best Val Acc: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/inside_pred/data/inclusion_train_mini.json")
    parser.add_argument("--image_dir", type=str, default="/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/data/train/images")
    parser.add_argument("--depth_dir", type=str, default="/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/data/train/depths")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU (default: 64)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers (default: 8)")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision for faster training")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum training samples to use (balanced between classes)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Maximum validation samples to use (balanced between classes)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="ckpt")
    parser.add_argument("--use_geometry", action="store_true", help="Use GeometryAwareInclusionModel with geometric features")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g., ckpt/latest.pth or ckpt/epoch_X.pth)")
    args = parser.parse_args()
    main(args)
