#!/bin/bash
# train_1_baseline.sh
DATA_ROOT="../hw4_2_data"
OUTPUT_DIR="./output/setting_1"
ITERATIONS=1000

echo "Starting Setting 1: Baseline..."

python train.py \
    -s ${DATA_ROOT} \
    -m ${OUTPUT_DIR} \
    --n_views 3 \
    --iterations ${ITERATIONS} \
    --pp_optimizer \
    --optim_pose \
    --save_iterations ${ITERATIONS} \
    --checkpoint_iterations ${ITERATIONS} \
    --densify_until_iter -1  # FORCE DISABLED