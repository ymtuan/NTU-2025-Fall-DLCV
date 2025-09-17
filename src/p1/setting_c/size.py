from PIL import Image

img = Image.open("../../../data_2025/p1_data/mini/train/206.jpg")
print(img.size)

img = Image.open("../../../data_2025/p1_data/office/train/1_1.jpg")
print(img.size)

img = Image.open("../../../data_2025/p1_data/office/val/0_32.jpg")
print(img.size)