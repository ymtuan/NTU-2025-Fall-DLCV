#!/bin/bash

# $1: path to the txt file containing the index
# $2: path to the original image pair directory
# $3: path to the interpolated sequence directory
# $4: path to the model checkpoint
# $5: path for the output prediction file

python dust3r_inference.py \
    --index_txt_path "$1" \
    --data_root "$2" \
    --interpolated_dir "$3" \
    --use_original_endpoints \
    --model_path "$4" \
    --save_pose_path "$5" \
    --gt_npy_path "hw4_1_data/public/gt.npy" \
    --output_dir "outputs/p1" \
    --seed 0 \
    --use_model "Dust3R"
