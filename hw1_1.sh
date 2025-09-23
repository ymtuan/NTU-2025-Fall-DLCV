#!/bin/bash

# TODO - run your inference Python3 code
MODEL_PATH="checkpoints/p1.pth"

python3 src/p1/inference.py "$1" "$2" "$3" "$MODEL_PATH"
