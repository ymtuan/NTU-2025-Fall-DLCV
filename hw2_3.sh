#!/bin/bash

# TODO - run your inference Python3 code
# HW2-3: ControlNet Inference Script
# Usage: bash hw2_3.sh $1 $2 $3 $4
# $1: path to the json file containing the testing conditions
# $2: path to the input image conditions folder
# $3: path to your output folder
# $4: path to the pretrained model weight

# Check if all arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: bash hw2_3.sh <json_path> <input_dir> <output_dir> <model_ckpt>"
    exit 1
fi

JSON_PATH=$1
INPUT_DIR=$2
OUTPUT_DIR=$3
MODEL_CKPT=$4

CONFIG_PATH_ABS=$(realpath "src/p3/ControlNet/models/cldm_v15.yaml")

# Print arguments for debugging
echo "JSON Path: $JSON_PATH"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Model Checkpoint: $MODEL_CKPT"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run inference
python3 src/p3/ControlNet/inference.py \
    --json_path "$JSON_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_ckpt "$MODEL_CKPT" \
    --config "$CONFIG_PATH_ABS" \


echo "Inference completed!"