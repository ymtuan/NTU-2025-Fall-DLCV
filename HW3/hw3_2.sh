#!/bin/bash

# $1: path to test images folder
# $2: path to output json file
# $3: path to decoder weights

python3 src/p2/inference.py "$1" "$2" "$3"