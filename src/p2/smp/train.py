import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex

from datasets import DeepGlobeDataset
from transforms import get_training_augmentation, get_validation_augmentation, get_preprocessing
from utils import get_loss, save_model

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "../../../data_2025/p2_data"
TRAIN_DIR = os.path.join(DATA_DIR, "train/")
VAL_DIR = os.path.join(DATA_DIR, "validation/")

ENCODER = "resnet50"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 7
ACTIVATION = None

EPOCHS = 200
BATCH_SIZE = 16  # reduce if OOM
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

criterion = get_loss()  # CrossEntropyLoss(ignore_index=6)
optimizer = optim.Adam(model.parameters(), lr=LR)

# IoU metric ignoring Unknown (class 6)
iou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=6).to(DEVICE)

# -----------------------------
# Training / Validation Loop
# -----------------------------
best_iou = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # --- Training ---
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc="Training"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

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
    iou_metric.reset()
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            iou_metric.update(preds, masks)
    val_loss /= len(val_loader)
    val_iou = iou_metric.compute().item()
    print(f"Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")

    # --- Save best model ---
    if val_iou > best_iou:
        best_iou = val_iou
        save_model(model, "best_deeplabv3plus.pth")
        print(f"New best model saved with mIoU: {best_iou:.4f}")
