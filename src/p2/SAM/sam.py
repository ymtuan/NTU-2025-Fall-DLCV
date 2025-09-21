import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import imageio
import numpy as np

# Load SAM model
sam_checkpopint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpopint)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# validation images
image_paths = [
    "validation/0018_sat.jpg",
    "validation/0065_sat.jpg",
    "validation/0109_sat.jpg"
]

for image_path in image_paths:
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to read {image_path}, skipping...")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image_rgb)

    mask_to_save = (masks[0]['segmentation'][:, :, np.newaxis] * 255).repeat(3, axis=2).astype(np.uint8)
    save_path = image_path.replace("_sat.jpg", "_sam.png")
    imageio.imsave(save_path, mask_to_save)