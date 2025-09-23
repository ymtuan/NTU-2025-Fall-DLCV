#!/bin/bash
# This script downloads checkpoint files for DLCV hw1 into checkpoints/ directory

# Download checkpoints using gdown

# Download checkpoint 1 for p1 setting c
gdown -id 1mo6eYM06XkR1nujhRKeHHQtT0XN0dHAN -O checkpoints/p1.pth

# Download checkpoint 2 for p2 model b (DeepLabV3+)
gdown -id 1UbP7W8Jtp5RC7YjyZGyuffXqge3VK6oY -O checkpoints/p2.pth

echo "All checkpoints download to checkpoints/"