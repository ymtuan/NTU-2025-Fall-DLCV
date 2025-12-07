# Run train_eval.sh 

### Conda
```bash
conda activate spatialagent
```

### Setup vllm
```bash
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --max-model-len 12288 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.7 \
    --port 8040
```

### Run on val set (size: 630)
```bash
cd SpatialAgent/agent
python train_eval.py --dataset val --limit 630 --verbose --llm_type vllm --api_base http://localhost:8040/v1 --model_name Qwen/Qwen3-4B-Instruct-2507
```

### Run on test set (size: 1312)
```bash
python train_eval.py --dataset test --limit 1312 --verbose --llm_type vllm --api_base http://localhost:8040/v1 --model_name Qwen/Qwen3-4B-Instruct-2507
```

---
