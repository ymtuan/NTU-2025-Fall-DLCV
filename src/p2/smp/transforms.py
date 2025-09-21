import torch
from torchvision import transforms
import random
import numpy as np

# -------------------------
# Data augmentations
# -------------------------
def get_training_augmentation():
    def augment(image, mask):
        # image: HWC numpy array, mask: HW numpy array

        # --- Flip ---
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # --- Small Rotation (±15°) ---
        angle = random.uniform(-15, 15)
        if abs(angle) > 1e-3:  # skip if angle ~0
            image = rotate_image(image, angle)
            mask = rotate_image(mask, angle, is_mask=True)

        # --- Random Color Jitter (brightness ±10%, contrast ±10%) ---
        image = color_jitter(image, brightness=0.1, contrast=0.1)

        return image, mask

    return augment


# -------------------------
# Helper functions
# -------------------------
from PIL import Image, ImageEnhance

def rotate_image(img, angle, is_mask=False):
    """Rotate numpy array by angle degrees"""
    mode = Image.NEAREST if is_mask else Image.BILINEAR
    pil_img = Image.fromarray(img)
    rotated = pil_img.rotate(angle, resample=mode)
    return np.array(rotated)

def color_jitter(img, brightness=0.1, contrast=0.1):
    """Apply mild brightness and contrast jitter"""
    pil_img = Image.fromarray(img)
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)
    return np.array(pil_img)


def get_validation_augmentation():
    def augment(image, mask):
        return image, mask
    return augment

# -------------------------
# Preprocessing
# -------------------------
def get_preprocessing(preprocess_fn=None):
    """
    preprocess_fn: optional function to normalize image (expects HWC format)
    Works for both training and inference.
    """
    def preprocess(image, mask):
        # Ensure numpy float32
        image = image.astype(np.float32)

        # If accidentally given CHW, convert back to HWC for preprocess_fn
        if image.ndim == 3 and image.shape[0] in (1, 3):
            # likely CHW -> convert to HWC
            image = np.transpose(image, (1, 2, 0))

        # Apply preprocessing (normalize, etc.)
        if preprocess_fn:
            image = preprocess_fn(image).astype(np.float32)

        # Convert back to tensor (CHW)
        image = torch.from_numpy(image.transpose(2, 0, 1))  # HWC -> CHW
        mask = torch.from_numpy(mask).long()
        return image, mask
    return preprocess
