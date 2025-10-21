#!/bin/bash

# TODO - run your inference Python3 code

# Check if all arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: bash hw2_2.sh <noise_dir> <output_dir> <model_weight>"
    exit 1
fi

NOISE_DIR=$1
OUTPUT_DIR=$2
MODEL_WEIGHT=$3

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the inference script
python src/p2/inference.py \
    --noise_path "$NOISE_DIR" \
    --output_path "$OUTPUT_DIR" \
    --ckpt_path "$MODEL_WEIGHT"