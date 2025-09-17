import numpy as np
from PIL import Image

mask = np.array(Image.open("../../../data_2025/p2_data/train/masks/1970_mask.png").convert("RGB"))
print(np.unique(mask.reshape(-1, 3), axis=0))