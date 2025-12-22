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

# dataset for distance_est
gdown --id 19vQntI6TV_PuLPB5rO8xCOoSl74J_Eaj \
      -O SpatialAgent/distance_est/train_distance_pairs.json

gdown --id 1t6pngvIM_k7FY0NbAfPk7cvHacWIfsGb \
      -O SpatialAgent/distance_est/val_distance_pairs.json

mkdir -p SpatialAgent/inside_pred/data
# dataset for inside_pred
gdown --id 1xO9-ykiYwqCccesJ_68AMZ0C7SvF4aYs \
      -O SpatialAgent/inside_pred/data/inclusion_train.json

gdown --id 10WegRb0A9WgQMM9ucFCK2OxCawvtkBW5 \
      -O SpatialAgent/inside_pred/data/inclusion_val.json


echo "All checkpoints downloaded to checkpoints/"
