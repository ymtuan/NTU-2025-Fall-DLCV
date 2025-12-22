#!/usr/bin/env bash

set -e

# Create checkpoints directory if it does not exist
mkdir -p checkpoints

echo "Downloading checkpoints..."

# log_mse.pth
gdown --id 1aSUQnor248KqgkRPJ4AYUThJuvauc9nO \
      -O checkpoints/log_mse.pth

# inclusion_model.pth
gdown --id 1yCRpdzKhBWqfwRjGmXerKyBbKDtfFNwP \
      -O checkpoints/inclusion_model.pth

# add_shortcut.pth
gdown --id 101a6u4dc6BD9xRPHoeLzcXnmgS6z2dv6 \
      -O checkpoints/add_shortcut.pth

echo "All checkpoints downloaded to checkpoints/"
