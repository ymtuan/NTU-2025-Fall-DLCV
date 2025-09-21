#!/bin/bash

EPOCHS=$1

if [ -z "$EPOCHS" ]; then
    echo "Usage: $0 <num_epochs>"
    exit 1
fi

python3 mean_iou_evaluate.py -g ../../../data_2025/p2_data/validation/ -p ./prediction/ > miou_${EPOCHS}epochs.log