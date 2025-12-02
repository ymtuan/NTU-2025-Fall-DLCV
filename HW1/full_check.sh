#!/bin/bash

pip3 install -r requirements.txt
pip3 install -r extra_requirements.txt

chmod +x get_dataset.sh hw1_download_ckpt.sh hw1_1.sh hw1_2.sh

bash get_dataset.sh || { echo "Dataset download failed"; exit 1; }
unzip -o -q data_2025.zip

mkdir checkpoints
bash hw1_download_ckpt.sh || { echo "Checkpoint download failed"; exit 1; }

mkdir -p results/p1
mkdir -p results/p2

bash hw1_1.sh ./data_2025/p1_data/office/val.csv ./data_2025/p1_data/office/val/ ./results/p1/p1_val_pred.csv
python3 p1_acc.py results/p1/p1_val_pred.csv data_2025/p1_data/office/val.csv 

bash hw1_2.sh data_2025/p2_data/validation/ results/p2
python3 mean_iou_evaluate.py -g data_2025/p2_data/validation/ -p results/p2/

echo "All inference and evaluation completed. Check the results/ directory."