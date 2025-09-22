import glob
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

from datasets import mask_to_rgb
from transforms import get_preprocessing
from utils import load_model

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "../../../data_2025/p2_data/validation"
OUTPUT_DIR = "report_mask/epoch1"
MODEL_PATH = "checkpoints/ce/best_deeplabv3plus_epoch1.pth"

ENCODER = "resnet50"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 7
ACTIVATION = None

# -----------------------------
# Model + Preprocessing
# -----------------------------
preprocess_input = smp.encoders.get_preprocessing_fn(
    ENCODER, pretrained=ENCODER_WEIGHTS
)

model = smp.DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
    activation=ACTIVATION,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
load_model(model, MODEL_PATH, DEVICE)
model.to(DEVICE)
model.eval()

preprocess = get_preprocessing(preprocess_input)

# -----------------------------
# Select only target samples
# -----------------------------
target_ids = {"0018", "0065", "0109"}

sat_files = [
    f for f in glob.glob(f"{DATA_DIR}/*_sat.jpg")
    if f.split("/")[-1].replace("_sat.jpg", "") in target_ids
]
sat_files.sort()

# -----------------------------
# Inference loop
# -----------------------------

for img_path in sat_files:
    sample_id = img_path.split("/")[-1].replace("_sat.jpg", "")

    # load image
    image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    mask_dummy = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # dummy mask

    # preprocessing (to tensor, normalize, etc.)
    image, _ = preprocess(image, mask_dummy)
    image = image.unsqueeze(0).to(DEVICE).float()  # ensure float32

    # prediction
    with torch.no_grad():
        output = model(image)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # convert to RGB mask
    pred_mask_rgb = mask_to_rgb(pred_mask)
    out_img = Image.fromarray(pred_mask_rgb)
    out_img.save(f"{OUTPUT_DIR}/{sample_id}_mask.png")
