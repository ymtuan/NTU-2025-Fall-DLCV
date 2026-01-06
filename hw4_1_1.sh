#!/bin/bash

# $1: path to the txt file containing the index
# $2: path to the original image pair directory
# $3: path to the model checkpoint
# $4: path for the output prediction file

python dust3r_inference.py \
    --index_txt_path "$1" \
    --data_root "$2" \
    --model_path "$3" \
    --save_pose_path "$4" \
    --gt_npy_path "" \
    --output_dir "outputs/p1" \
    --test_only \
    --seed 123 \
    --use_model "Dust3R"
