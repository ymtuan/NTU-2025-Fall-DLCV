#!/bin/bash

# ==========================================
# TRAINING SCRIPT (Local Use Only)
# ==========================================

# 1. Path Definitions
# Point this to where you downloaded the data (the folder containing sparse_3)
DATA_ROOT="../hw4_2_data" 
# Where to save the trained model
OUTPUT_DIR="./output"

# 2. Hyperparameters (You need to experiment with these!)
ITERATIONS=1000  # Try 1000, 2000, etc.
N_VIEWS=3        # Do not change, dataset is sparse_3

echo "Starting Training on ${DATA_ROOT}..."

# 3. Run Training
# We SKIP init_geo.py because the data provided is already initialized.
# We use --pp_optimizer as per InstantSplat defaults for sparse views.
python train.py \
    -s ${DATA_ROOT} \
    -m ${OUTPUT_DIR} \
    --n_views ${N_VIEWS} \
    --iterations ${ITERATIONS} \
    --pp_optimizer \
    --optim_pose \
    --save_iterations ${ITERATIONS} \
    --checkpoint_iterations ${ITERATIONS}

echo "Training Complete. Model saved to ${OUTPUT_DIR}"