#!/bin/bash
# This script downloads checkpoint files for DLCV hw2 into checkpoints/

# Download checkpoints using gdown

mkdir -p checkpoints

# Download checkpoint 1 for p1
gdown -id 1NKeoUBxCSTDPSM_R9ZF9wSQdkAUf7FCy -O checkpoints/p1.pth

# Download checkpoint 2 for p2 (pretrained UNet.pt)
gdown -id 1o6y_DgxglZOBfb1sLBZYqssrtdRXehet -O checkpoints/p2.pt

# Download checkpoint 3 for p3
gdown -id 17hwHtYjHMu124TS-Vi3m3wM1x3hPYT2I -O checkpoints/p3.ckpt