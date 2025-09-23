#!/bin/bash

# TODO - run your inference Python3 code
python3 src/p2/smp/inference.py --input_dir="$1" --output_dir="$2" --model_path checkpoints/p2.pth