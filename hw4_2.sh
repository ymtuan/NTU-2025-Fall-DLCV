#!/bin/bash

# $1: Path to the folder containing test data (e.g. hw4/p2_data/)
# $2: Path to the output png files

MODEL_PATH="./p2_checkpoints/setting_2" 
LOAD_ITER=1000 

python3 InstantSplat/render_p2.py \
    -s "$1" \
    -m "$MODEL_PATH" \
    --output_path "$2" \
    --iteration $LOAD_ITER \
    --n_views 3 \
    --eval \
    --quiet