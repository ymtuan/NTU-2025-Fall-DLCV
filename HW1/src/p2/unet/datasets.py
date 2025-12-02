import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

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
        # collect all *_sat.jpg files
        sat_files = glob.glob(f"{images_dir}/*_sat.jpg")
        self.ids = [f.split("/")[-1].replace("_sat.jpg", "") for f in sat_files]
        self.ids.sort()

        self.images_dir = images_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        img_path = f"{self.images_dir}/{sample_id}_sat.jpg"
        mask_path = f"{self.images_dir}/{sample_id}_mask.png"

        # use PIL instead of cv2
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)

        mask = rgb_to_mask(mask_rgb)

        if self.augmentation:
            image, mask = self.augmentation(image, mask)

        if self.preprocessing:
            image, mask = self.preprocessing(image, mask)

        mask = mask.long()  # final cast to long tensor

        return image, mask

    def __len__(self):
        return len(self.ids)
