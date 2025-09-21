import os
import cv2
import numpy as np
from torch.utils.data import Dataset

COLOR2CLASS = {
    (0, 255, 255): 0,   # Cyan Urban
    (255, 255, 0): 1,   # Yellow Agriculture
    (255, 0, 255): 2,   # Purple Rangeland
    (0, 255, 0): 3,     # Green Forest
    (0, 0, 255): 4,     # Blue Water
    (255, 255, 255): 5, # White Barren
    (0, 0, 0): 6,       # Black Unknown
}
CLASS2COLOR = {v: k for k, v in COLOR2CLASS.items()}


def rgb_to_mask(mask_rgb):
    h, w, _ = mask_rgb.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, cls in COLOR2CLASS.items():
        mask[(mask_rgb == rgb).all(axis=-1)] = cls
    return mask


def mask_to_rgb(mask):
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in CLASS2COLOR.items():
        mask_rgb[mask == cls] = rgb
    return mask_rgb


class DeepGlobeDataset(Dataset):
    def __init__(self, images_dir, augmentation=None, preprocessing=None):
        self.images_dir = images_dir
        # collect all *_sat.jpg files
        self.ids = [f.replace("_sat.jpg", "") for f in os.listdir(images_dir) if f.endswith("_sat.jpg")]
        self.ids.sort()
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        img_path = os.path.join(self.images_dir, f"{sample_id}_sat.jpg")
        mask_path = os.path.join(self.images_dir, f"{sample_id}_mask.png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_rgb = cv2.imread(mask_path)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask = rgb_to_mask(mask_rgb)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        mask = mask.long()  # <-- important

        return image, mask


    def __len__(self):
        return len(self.ids)
