#!/bin/bash

nohup python3 finetune_d.py "../../../data_2025/p1_data/office/train.csv" "../../../data_2025/p1_data/office/train" "../../../data_2025/p1_data/office/val.csv" "../../../data_2025/p1_data/office/val" > output_d.log 2>&1 &