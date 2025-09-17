#!/bin/bash
# Script to launch DINO training on 1×L40s GPU using nohup

# Activate your Python environment if needed
# Example:
# source ~/.pyenv/versions/3.8.20/bin/activate

echo 'Starting DINO training on L40s with batch=128, epochs=500...'

# Run training in background with nohup
nohup python main_dino.py \
  --arch resnet50 \
  --epochs 500 \
  --optimizer sgd \
  --lr 0.03 \
  --weight_decay 1e-4 \
  --weight_decay_end 1e-6 \
  --warmup_epochs 10 \
  --global_crops_scale 0.14 1 \
  --local_crops_scale 0.05 0.14 \
  --local_crops_number 6 \
  --batch_size_per_gpu 128 \
  --saveckp_freq 20 \
  --data_path ../data_2025/p1_data/mini/train \
  --output_dir setting_c/ \
  > setting_c/train.log 2>&1 &

echo 'Training started in background. Logs are being written to setting_c/train.log'
echo 'You can close the SSH terminal; the process will continue running.'
echo 'To monitor progress, use: tail -f setting_c/train.log'