import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import segmentation_models_pytorch as smp

from datasets import DeepGlobeDataset
from transforms import get_training_augmentation, get_validation_augmentation, get_preprocessing
from utils import save_model, mean_iou_score, FocalLoss, IoULoss

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "../../../data_2025/p2_data"
TRAIN_DIR = f"{DATA_DIR}/train/"
VAL_DIR = f"{DATA_DIR}/validation/"
SAVE_DIR = "checkpoints"

ENCODER = "resnet50"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 7
ACTIVATION = None

EPOCHS = 400
BATCH_SIZE = 16
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Print hyperparameters
# -----------------------------
print("===== Training Hyperparameters =====")
print(f"Encoder: {ENCODER}, Pretrained: {ENCODER_WEIGHTS}")
print(f"Num Classes: {NUM_CLASSES}, Activation: {ACTIVATION}")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LR}")
print(f"Device: {DEVICE}")
print("===================================")

# -----------------------------
# Dataset and DataLoader
# -----------------------------
preprocess_input = smp.encoders.get_preprocessing_fn(ENCODER, pretrained=ENCODER_WEIGHTS)

train_dataset = DeepGlobeDataset(
    images_dir=TRAIN_DIR,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

val_dataset = DeepGlobeDataset(
    images_dir=VAL_DIR,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,
    classes=NUM_CLASSES,
    activation=ACTIVATION
)
model.to(DEVICE)

# criterion = nn.CrossEntropyLoss(ignore_index=6)
# criterion = FocalLoss(gamma=2.0, ignore_index=6)
criterion = IoULoss(ignore_index=6)

optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Cosine Annealing Scheduler
# -----------------------------
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# -----------------------------
# Training / Validation Loop
# -----------------------------
best_miou = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # --- Training ---
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc="Training"):
        images = images.to(DEVICE).float()
        masks = masks.to(DEVICE).long()

        optimizer.zero_grad()
        outputs = model(images)  # (B, C, H, W)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Train Loss: {train_loss:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_masks.append(masks.cpu().numpy())

    val_loss /= len(val_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    val_miou = mean_iou_score(all_preds, all_masks)

    print(f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

    # --- Save best model by mIoU ---
    if val_miou > best_miou:
        best_miou = val_miou
        save_model(model, f"checkpoints/jaccard/best_deeplabv3plus_epoch{epoch+1}.pth")
        print(f"New best model saved with mIoU: {best_miou:.4f}")