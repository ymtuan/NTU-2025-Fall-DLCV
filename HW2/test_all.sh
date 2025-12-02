#!/bin/bash

# conda create -y -n dlcv_hw2 python=3.8.5
# conda activate dlcv_hw2

bash hw2_download_ckpt.sh

# Download test dataset and unzip
gdown 1-jBoXNaW5hcb8yvSRmiWILPZnjeiXBxP -O test_data.zip
unzip ./test_data.zip -d ./

mkdir -p outputs

bash hw2_1.sh outputs/p1

bash hw2_2.sh test_data/face/noise outputs/p2 checkpoints/p2.pt

cd stable-diffusion
pip install -e .
cd ..

bash hw2_3.sh test_data/fill50k/testing/prompt.json test_data/fill50k/testing/source outputs/p3 checkpoints/p3.ckpt