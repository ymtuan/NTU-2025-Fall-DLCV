import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
import imageio
import numpy as np

# Load SAM model
sam_checkpopint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpopint)
sam.to(device)

predictor = SamPredictor(sam)

# validation images
image_paths = [
    "validation/0018_sat.jpg",
    "validation/0065_sat.jpg",
    "validation/0109_sat.jpg"
]

for image_path in image_paths:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    masks, scores, logits = predictor.predict(multimask_output=True)

    mask_to_save = (masks[0] * 255).astype(np.uint8)
    save_path = image_path.replace(".jpg", "_mask.png")
    imageio.imsave(save_path, mask_to_save)