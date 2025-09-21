import os
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from datasets import mask_to_rgb
from utils import load_model

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "../../../data_2025/p2_data/validation/"
OUT_DIR = "prediction"
os.makedirs(OUT_DIR, exist_ok=True)

ENCODER = "resnet50"
NUM_CLASSES = 7

# -----------------------------
# Build model
# -----------------------------
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(model, "best_deeplabv3plus_epoch108.pth", DEVICE)
model.to(DEVICE)
model.eval()

# -----------------------------
# Preprocessing
# -----------------------------
preprocess_input = smp.encoders.get_preprocessing_fn(ENCODER, pretrained="imagenet")
transform = A.Compose([
    A.Lambda(image=lambda x, **kwargs: preprocess_input(x).astype("float32")),
    ToTensorV2()
])

# -----------------------------
# Run inference
# -----------------------------
with torch.no_grad():
    for file in sorted(os.listdir(DATA_DIR)):
        if not file.endswith("_sat.jpg"):
            continue

        img_path = os.path.join(DATA_DIR, file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess
        tensor = transform(image=image)["image"].unsqueeze(0).to(DEVICE)  # shape: (1,3,H,W)

        # predict
        output = model(tensor)  # (1, NUM_CLASSES, H, W)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # (H, W)

        # convert to RGB
        mask_rgb = mask_to_rgb(mask)

        # save mask as PNG with corresponding name
        out_name = file.replace("_sat.jpg", "_mask.png")
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

print(f"Predictions saved to {OUT_DIR}")
