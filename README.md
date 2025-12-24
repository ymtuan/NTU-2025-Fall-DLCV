# DLCV_1141_final_challenge_1

This repository implements Challenge 1 for 114-1 DLCV final project Warehouse Spatial Intelligence (Track 3 of Nvidia AI City Challenge in ICCV 2025)
By analyzing the framework proposed in [this paper](https://arxiv.org/abs/2507.10778), we introduce a two-stage curriculum training strategy for the distance estimation model, achieving near-perfect precision on the validation set.
In addition, through self-curated dataset construction, we trained an inclusion classification model that improved validation accuracy from approximately 5% to over 90%.

Overall, these contributions earned third place in the final project presentation.

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
pip install flash-attn --no-build-isolation
```

### 3. Download checkpoints
```bash
bash get_ckpt.sh
```

### 4. Download dataset
```bash
git clone https://huggingface.co/datasets/yaguchi27/DLCV_Final1
cd DLCV_Final1
tar -xvf train/images/images.tar.gz -C train/images/
tar -xvf val/images/images.tar.gz -C val/images/
tar -xvf test/images/images.tar.gz -C test/images/
tar -xvf train/depths/depths.tar.gz -C train/depths/
tar -xvf val/depths/depths.tar.gz -C val/depths/
tar -xvf test/depths/depths.tar.gz -C test/depths/
rm train/images/images.tar.gz val/images/images.tar.gz test/images/images.tar.gz train/depths/depths.tar.gz val/depths/depths.tar.gz test/depths/depths.tar.gz
cd ..

# Move data to SpatialAgent/data directory
# mv DLCV_Final1/train SpatialAgent/data/
# mv DLCV_Final1/val SpatialAgent/data/
# mv DLCV_Final1/test SpatialAgent/data/
# mv DLCV_Final1/train.json SpatialAgent/data/
# mv DLCV_Final1/val.json SpatialAgent/data/
# mv DLCV_Final1/test.json SpatialAgent/data/
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

## Training 

### distance model training
```bash
cd ./SpatialAgent/distance_est
python train.py --use_geometry --use_shortcut --pretrained --data_dir ../../DLCV_Final1/train --train_json train_distance_pairs.json --val_json val_distance_pairs.json
```

### inclusion model training
```bash
cd ./SpatialAgent/inside_pred
python train.py --use_geometry --use_soft_labels --hard_sample_weighting --aux_loss_weight 0.5  --json data/inclusion_train.json --image_dir ../../DLCV_Final1/train/images/ --depth_dir ../../DLCV_Final1/train/depths/
```
