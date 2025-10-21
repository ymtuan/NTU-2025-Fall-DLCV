#!/bin/bash

# TODO - run your inference Python3 code
# Check if output directory argument is provided
if [ -z "$1" ]; then
    echo "Error: Please provide output directory path"
    echo "Usage: bash hw2_1.sh <output_directory>"
    exit 1
fi

# Set output directory
OUTPUT_DIR=$1

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run inference script to generate 500 images
python3 src/p1/inference.py \
    --output_image_dir "$OUTPUT_DIR" \
    --model_path src/p1/checkpoints/model_199.pth \

echo "Generated 500 images saved to: $OUTPUT_DIR"