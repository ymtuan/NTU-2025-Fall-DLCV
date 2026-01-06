#!/bin/bash
# densify.sh
DATA_ROOT="../hw4_2_data"
OUTPUT_DIR="./output/setting_4"
ITERATIONS=1000

echo "Starting Setting 4: Standard Densification..."

python train.py \
    -s ${DATA_ROOT} \
    -m ${OUTPUT_DIR} \
    --n_views 3 \
    --iterations ${ITERATIONS} \
    --optim_pose \
    --save_iterations ${ITERATIONS} \
    --checkpoint_iterations ${ITERATIONS} \
    --densify_from_iter 100 \
    --densify_until_iter 900 \
    --densification_interval 100 \
    --densify_grad_threshold 0.0002 # Standard