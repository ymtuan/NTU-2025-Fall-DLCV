#!/bin/bash
# high lr
DATA_ROOT="../hw4_2_data"
OUTPUT_DIR="./output/setting_2"
ITERATIONS=1000

echo "Starting Setting 2: High Learning Rate..."

python train.py \
    -s ${DATA_ROOT} \
    -m ${OUTPUT_DIR} \
    --n_views 3 \
    --iterations ${ITERATIONS} \
    --pp_optimizer \
    --optim_pose \
    --save_iterations ${ITERATIONS} \
    --checkpoint_iterations ${ITERATIONS} \
    --densify_until_iter -1 \
    --position_lr_init 0.00032 \
    --scaling_lr 0.01