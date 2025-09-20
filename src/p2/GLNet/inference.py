import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from helper import create_model_load_weights, images_transform, masks_transform
from dataset.deep_globe import DeepGlobe, classToRGB, is_image_file

# Setting
n_class = 7
mode = 3
evaluation = True

# Path to checkpoints
path_g="saved_models/fpn_deepglobe_global.pth"
path_g2l="saved_models/fpn_deepglobe_global2local.pth"
path_l2g="saved_models/fpn_deepglobe_local2global.pth"

# Validation dataset folder
val_root = "../../../data_2025/p2_data/validation"
val_ids = [f for f in os.listdir(val_root) if f.endswith("_sat.jpg")]

# Where to save predicitons
save_dir = "prediction"

# Preprocessing for inference
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Colormap
cls_color = {
    0: [0, 255, 255],   # urban
    1: [255, 255, 0],   # agriculture
    2: [255, 0, 255],   # rangeland
    3: [0, 255, 0],     # forest
    4: [0, 0, 255],     # water
    5: [255, 255, 255], # barren
    6: [0, 0, 0],       # unknown
}

def colorize_mask(mask, cls_color):
    """Convert class-indexed mask to RGB color image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in cls_color.items():
        color_mask[mask == cls_id] = color
    return color_mask


def main():
    model, global_fixed = create_model_load_weights(
        n_class=n_class,
        mode=mode,
        evaluation=evaluation,
        path_g=path_g,
        path_g2l=path_g2l,
        path_l2g=path_l2g
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    val_dataset = DeepGlobe(root=val_root, ids=val_ids, label=False, transform=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    with torch.no_grad():
        for sample in val_loader:
            img_id = sample['id']

            image = val_transform(sample['image']).unsqueeze(0).to(device)

            if global_fixed is not None:
                pred = model(image, global_fixed)
            else:
                pred = model(image)

            pred_label = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

            # Save raw mask
            raw_path = os.path.join(save_dir, f"{img_id}_pred_raw.png")
            Image.fromarray(pred_label.astype(np.uint8)).save(raw_path)

            # Save colorized mask
            color_mask = colorize_mask(pred_label, cls_color)
            color_path = os.path.join(save_dir, f"{img_id}_pred_color.png")
            Image.fromarray(color_mask).save(color_path)

            print(f"Saved: {raw_path} and {color_path}")

if __name__ == "__main__":
    main()