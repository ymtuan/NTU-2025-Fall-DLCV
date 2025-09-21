import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

def get_validation_augmentation():
    return A.Compose([])  # no augmentation

def get_preprocessing(preprocessing_fn):
    return A.Compose([
        A.Lambda(image=lambda x, **kwargs: preprocessing_fn(x).astype(np.float32)),
        ToTensorV2()
    ])