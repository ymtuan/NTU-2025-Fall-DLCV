#!/bin/bash
# #######################################  UPDATE THIS BLOCK ONLY #######################################

# --- 1. SET YOUR PATHS HERE ---
# Path to the index .txt file (defines pairs to process)
INDEX_TXT_PATH="/home/ymtuan/dl/dlcv-fall-2025-hw4-ymtuan/hw4_1_data/public_yaw_80.0_to_90.0.txt"

# Path to the ground truth .npy file
GT_NPY_PATH=/home/ymtuan/dl/dlcv-fall-2025-hw4-ymtuan/hw4_1_data/public/gt.npy

# Path to the .npy file you saved from the inference script
PRED_POSE_PATH=/home/ymtuan/dl/dlcv-fall-2025-hw4-ymtuan/outputs/p1/public/video_predict.npy
# PRED_POSE_PATH=/home/ymtuan/dl/dlcv-fall-2025-hw4-ymtuan/outputs/p1/video_dust3r_predicted_poses.npy

# #######################################  UPDATE THIS BLOCK ONLY #######################################


# --- 2. SET OPTIONS ---
# Which metrics to calculate ('R', 'T', 'both')
EVAL_MODE='R'

# --- 3. EXECUTE THE SCRIPT ---
echo "Running metric check..."

python eval.py \
    --index_txt_path "${INDEX_TXT_PATH}" \
    --gt_npy_path "${GT_NPY_PATH}" \
    --pred_pose_path "${PRED_POSE_PATH}" \
    --eval_mode ${EVAL_MODE}

echo "Metric check finished."