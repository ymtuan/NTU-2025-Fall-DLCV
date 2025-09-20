import random
import os

image_folder = "../../../data_2025/p2_data/validation"
all_images = [f for f in os.listdir(image_folder) if f.endswith("_sat.jpg")]

random.shuffle(all_images)

with open("train.txt", "w") as f:
    for img_name in all_images:
        f.write(img_name + "\n")