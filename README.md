# DLCV_1141_final_challenge_1

This repository implements Challenge 1 for 114-1 DLCV final project Warehouse Spatial Intelligence (Track 3 of Nvidia AI City Challenge in ICCV 2025)


## Environemnt Setup

### 1. Clone the repository
```bash
git clone git@github.com:snooow1029/DLCV_1141_final_challenge_1.git
cd DLCV_1141_final_challenge_1
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate DLCV_Final_Challenge_1
```

### 3. Download checkpoints
```bash
bash get_ckpt.sh
```

### 4. Download dataset
```bash
git clone https://huggingface.co/datasets/yaguchi27/DLCV_Final1
cd DLCV_Final1
tar -xvf train/images.tar.gz 
tar -xvf test/images.tar.gz
cd ..
```

### 5. Setup VLLM
```bash
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --max-model-len 12288 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.7 \
    --port 8040
```

### 6. Start inference on test set
```bash
cd SpatialAgent/agent
python train_eval.py \
    --dataset test \
    --limit 1312 \
    --verbose \
    --llm_type vllm \
    --api_base http://localhost:8040/v1 \
    --model_name Qwen/Qwen3-4B-Instruct-2507
```

### 7. Check prediction results
```bash
cd ../output
```
prediction file will be saved as predictions.json