#!/bin/bash

# Usage: bash hw3_1.sh <annotation_json> <images_root> <llava_weight_path> <output_json>
# We run the VCD inference script with a 30-minute timeout to respect time limits.
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <annotation_json> <images_root> <llava_weight_path> <output_json>"
  exit 2
fi

ANNOTATION="$1"
IMAGES_ROOT="$2"
LLAVA_PATH="$3"
OUTPUT="$4"

python3 src/p1/p1_2_inference.py "$ANNOTATION" "$IMAGES_ROOT" "$LLAVA_PATH" "$OUTPUT"