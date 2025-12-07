#!/bin/bash

# Check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown not found."
fi

python -m gdown --id 19-CN0ZjEBM0LjHgmMmDlDhguWJQzZT2y -O distance_est/ckpt/3m_epoch6.pth

python -m gdown --id 1kV0iLpkU_cpujzverm9f5JOPSZfE6kAE -O distance_est/ckpt/epoch_5_iter_6831.pth

python -m gdown --id 1NRlHIkoCi5upQNufZJbN3AQx8mGKgHSw -O inside_pred/ckpt/epoch_4.pth

# dataset
python -m gdown --id 1ENe42GsqxuP9GbOT64HlmbSfq70mYeg_ -O data.zip
unzip -o data.zip
rm data.zip

echo "Download finished."