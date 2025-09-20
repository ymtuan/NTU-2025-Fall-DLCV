from PIL import Image
import numpy as np
m = np.array(Image.open("../../../data_2025/p2_data/train/0322_mask.png"))
print(np.unique(m))

