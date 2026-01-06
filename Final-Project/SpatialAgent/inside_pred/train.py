import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from data_loader import InsideDataset
from model import ResNet50Binary, GeometryAwareInclusionModel
from loss import FocalLoss
import os

def evaluate(model, loader, criterion, device, use_geometry=False, aux_loss_weight=0.5):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    edge_case_correct = 0
    edge_case_total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            if use_geometry:
                inputs, geo_features, labels = batch
                inputs = inputs.to(device, non_blocking=True)
                geo_features = geo_features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # 提取 IoU 用于 edge case 统计
                iou_values = geo_features[:, 0]
                edge_case_mask = (iou_values >= 0.3) & (iou_values <= 0.7)
                
                main_outputs, geo_outputs = model(inputs, geo_features)
                
                # 计算总 loss（主 loss + 辅助 loss）
                loss_main = criterion(main_outputs, labels)
                loss_geo = criterion(geo_outputs, labels)
                loss = loss_main + aux_loss_weight * loss_geo
                
                # 使用主输出进行预测
                outputs = main_outputs
            else:
                inputs, labels = batch
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                edge_case_mask = None
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Edge case 准确率统计
            if edge_case_mask is not None:
                edge_case_preds = preds[edge_case_mask]
                edge_case_labels = labels[edge_case_mask]
                edge_case_correct += (edge_case_preds == edge_case_labels).sum().item()
                edge_case_total += edge_case_mask.sum().item()
    
    loss = running_loss / total
    acc = correct / total
    edge_case_acc = edge_case_correct / edge_case_total if edge_case_total > 0 else 0.0
    return loss, acc, edge_case_acc

def train_one_epoch(model, loader, optimizer, criterion, device, use_geometry=False, scaler=None, accumulation_steps=1, label_smoothing=0.0, use_hard_sample_weighting=False, aux_loss_weight=0.5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    hard_sample_count = 0  # 统计难样本数量

    pbar = tqdm(loader, desc="Training", leave=False)
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        if use_geometry:
            inputs, geo_features, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            geo_features = geo_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 从几何特征中提取 IoU（第一个特征）
            iou_values = geo_features[:, 0]  # [B]
        else:
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            iou_values = None
        
        # --- Label Smoothing (可選) ---
        # 如果模型對 Hard Negatives 反應過度，開啟這個可以讓它不要那麼"自信"
        # labels 0 -> 0.05, 1 -> 0.95
        if label_smoothing > 0:
            target_labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            target_labels = labels

        # Mixed precision training
        with autocast(enabled=(scaler is not None)):
            if use_geometry:
                main_outputs, geo_outputs = model(inputs, geo_features)
                
                # 主 Loss (Fusion Head)
                if use_hard_sample_weighting and iou_values is not None:
                    # 使用 reduction='none' 获取每个样本的 loss
                    criterion_none = FocalLoss(alpha=criterion.alpha, gamma=criterion.gamma, reduction='none')
                    loss_main_per_sample = criterion_none(main_outputs, target_labels)  # [B]
                    
                    # IoU 在 0.3-0.7 之间是 edge case，给予更高权重
                    edge_case_mask = (iou_values >= 0.3) & (iou_values <= 0.7)
                    weights = torch.ones_like(loss_main_per_sample)
                    weights[edge_case_mask] = 2.0  # Edge case 权重翻倍
                    loss_main = (loss_main_per_sample * weights).mean()  # 加权后平均
                    hard_sample_count += edge_case_mask.sum().item()
                else:
                    loss_main = criterion(main_outputs, target_labels)
                
                # 輔助 Loss (Geometric Stream) - 強制幾何流也要準確預測
                loss_geo = criterion(geo_outputs, target_labels)
                
                # 總 Loss
                loss = loss_main + aux_loss_weight * loss_geo
                
                # 使用主输出进行预测
                outputs = main_outputs
            else:
                outputs = model(inputs)
                loss = criterion(outputs, target_labels)
            
            loss = loss / accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
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
        hard_ratio = 100 * hard_sample_count / total if total > 0 else 0
        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}", 
            "acc": f"{acc:.2f}%",
            "hard": f"{hard_ratio:.1f}%" if use_hard_sample_weighting else ""
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def get_val_path(path):
    return path.replace("train", "val")

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training dataset
    dataset = InsideDataset(
        json_path=args.json,
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        use_depth=False,
        use_geometry=args.use_geometry,
        max_samples=args.max_samples,
        use_soft_labels=args.use_soft_labels  # 训练时使用软标签
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

    # Validation dataset
    val_json = get_val_path(args.json)
    val_image_dir = get_val_path(args.image_dir)
    val_depth_dir = get_val_path(args.depth_dir)
    val_dataset = InsideDataset(
        json_path=val_json,
        image_dir=val_image_dir,
        depth_dir=val_depth_dir,
        use_depth=False,
        use_geometry=args.use_geometry,
        max_samples=args.max_val_samples,
        use_soft_labels=False  # 验证时使用硬标签
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2,
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

    # Resume & Freeze Logic
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                print("Detected old checkpoint format. Loading weights...")
                state_dict = checkpoint
            
            # 处理旧模型（没有 geo_aux_head）的情况
            if args.use_geometry and 'geo_aux_head.weight' not in state_dict:
                print("⚠️  Detected old model without geo_aux_head. Initializing auxiliary head...")
                # 初始化 geo_aux_head 的权重
                model.load_state_dict(state_dict, strict=False)
                # geo_aux_head 会使用默认初始化
            else:
                model.load_state_dict(state_dict)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                print(f"✓ Resumed from epoch {start_epoch}, best_val_acc: {best_val_acc*100:.2f}%")
        else:
            print(f"Warning: Checkpoint {args.resume} not found.")

    # ===== 關鍵修改：凍結 Backbone =====
    if args.freeze_backbone and args.use_geometry:
        print("\n❄️ FREEZING ResNet BACKBONE ❄️")
        print("Only training 'geo_mlp' (MLP) and 'fusion_head'.")
        
        # 鎖住 Visual Stream (ResNet)
        for param in model.resnet.parameters():
            param.requires_grad = False
            
        # 確保 Geometric Stream 和 Fusion Head 是開啟的
        for param in model.geo_mlp.parameters():
            param.requires_grad = True
        for param in model.fusion_head.parameters():
            param.requires_grad = True
            
        # 檢查一下 train 參數數量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable Params: {trainable_params:,} / {total_params:,}\n")

    # Optimizer (只傳入 requires_grad=True 的參數)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr, weight_decay=args.weight_decay)
    
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    scaler = GradScaler() if args.use_amp else None
    
    # Learning rate scheduler for better convergence
    if args.use_lr_scheduler:
        if args.lr_scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        elif args.lr_scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        else:
            scheduler = None
    else:
        scheduler = None

    os.makedirs(args.save_path, exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            args.use_geometry, scaler, args.accumulation_steps,
            label_smoothing=args.label_smoothing,
            use_hard_sample_weighting=args.hard_sample_weighting,
            aux_loss_weight=args.aux_loss_weight
        )
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
        val_loss, val_acc, val_edge_acc = evaluate(model, val_loader, criterion, device, args.use_geometry, args.aux_loss_weight)
        print(f"Epoch {epoch+1}/{args.epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Edge Case Acc: {val_edge_acc*100:.2f}%")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if args.lr_scheduler_type == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Current LR: {current_lr:.2e}")
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(args)
        }
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        # Save epoch checkpoint
        torch.save(checkpoint, f"{args.save_path}/epoch_{epoch+1}.pth")
        torch.save(checkpoint, f"{args.save_path}/latest.pth")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.save_path}/best_model.pth")
            print(f"✓ New best model saved! Val Acc: {val_acc*100:.2f}%")

    print(f"\nTraining completed. Best Val Acc: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (前面的參數保持不變) ...
    parser.add_argument("--json", type=str, default="data/inclusion_train.json")
    parser.add_argument("--image_dir", type=str, default="../../DLCV_Final1/train/images/")
    parser.add_argument("--depth_dir", type=str, default="../../DLCV_Final1/train/depths/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="ckpt")
    parser.add_argument("--use_geometry", action="store_true")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--resume", type=str, default=None)
    
    # 新增參數
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze ResNet weights and only train MLP head")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor (e.g., 0.1)")
    parser.add_argument("--hard_sample_weighting", action="store_true", default=False, help="Weight edge case samples (IoU 0.3-0.7) more heavily")
    parser.add_argument("--use_soft_labels", action="store_true", default=False, help="Use soft labels for edge cases (IoU 0.3-0.7) during training")
    parser.add_argument("--aux_loss_weight", type=float, default=0.5, help="Weight for auxiliary geometric loss (default: 0.5)")
    parser.add_argument("--use_lr_scheduler", action="store_true", default=False, help="Use learning rate scheduler")
    parser.add_argument("--lr_scheduler_type", type=str, default='cosine', choices=['cosine', 'plateau'], help="LR scheduler type: cosine or plateau")
    
    args = parser.parse_args()
    main(args)