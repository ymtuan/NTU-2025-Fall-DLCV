import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

from datasets import mask_to_rgb
from transforms import get_preprocessing
from utils import load_model

def main():
    parser = argparse.ArgumentParser(description="Segmentation inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of input satellite images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predicted masks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Model config
    # -----------------------------
    ENCODER = "resnet50"
    NUM_CLASSES = 7
    ACTIVATION = None

    preprocess_input = smp.encoders.get_preprocessing_fn(ENCODER, pretrained=None)
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=ACTIVATION,
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    load_model(model, args.model_path, DEVICE)
    model.to(DEVICE)
    model.eval()

    preprocess = get_preprocessing(preprocess_input)

    # -----------------------------
    # Inference loop
    # -----------------------------
    sat_files = sorted(input_dir.glob("*_sat.jpg"))

    for img_path in sat_files:
        sample_id = img_path.stem.replace("_sat", "")

        # load image
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask_dummy = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # preprocessing (to tensor, normalize, etc.)
        image, _ = preprocess(image, mask_dummy)
        image = image.unsqueeze(0).to(DEVICE).float()

        # prediction
        with torch.no_grad():
            output = model(image)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # convert to RGB mask
        pred_mask_rgb = mask_to_rgb(pred_mask)
        out_img = Image.fromarray(pred_mask_rgb)
        out_img.save(output_dir / f"{sample_id}_mask.png")

if __name__ == "__main__":
    main()
