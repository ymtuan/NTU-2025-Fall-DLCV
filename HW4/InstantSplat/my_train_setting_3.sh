#!/bin/bash
# long iter
DATA_ROOT="../hw4_2_data"
OUTPUT_DIR="./output/setting_3"
ITERATIONS=2000

echo "Starting Setting 3: Long Training..."

python train.py \
    -s ${DATA_ROOT} \
    -m ${OUTPUT_DIR} \
    --n_views 3 \
    --iterations ${ITERATIONS} \
    --pp_optimizer \
    --optim_pose \
    --save_iterations ${ITERATIONS} \
    --checkpoint_iterations ${ITERATIONS} \
    --densify_until_iter -1