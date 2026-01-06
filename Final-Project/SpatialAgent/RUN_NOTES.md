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

### With new prompt
python train_eval.py --dataset val --limit 300 --verbose --llm_type vllm --api_base http://localhost:8040/v1 --model_name Qwen/Qwen3-4B-Instruct-2507

dist_model_cfg={
    # 'model_path': '/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/agent/ckpt_log_mse/best_model.pth',
    'model_path': '/home/ymtuan/dl/DLCV_1141_final_challenge_1/SpatialAgent/distance_est/ckpt_log_mse/log_mse_best_model.pth',
    'use_geometry': True,      # 使用 geometric features
    'input_channels': 6,       # RGB(3) + Depth(1) + Mask1(1) + Mask2(1)
    'num_geo_features': 14      
},

準確率報告 (val set):
嚴格模式 (±0.1): 77.00% (231/300)
10% 容錯模式: 87.00% (261/300)
20% 容錯模式: 88.33% (265/300)
預設使用: 10% 容錯模式

問題分類準確率: 100.00% (300/300)

分類準確率（按類別）:
  left_right: 100.00% (84/84)
  count: 100.00% (74/74)
  distance: 100.00% (63/63)
  mcq: 100.00% (79/79)

錯誤分析 (val set):
總錯誤數: 39

按類別分布:
  left_right: 13 個錯誤
  count: 15 個錯誤
  mcq: 4 個錯誤
  distance: 7 個錯誤

常見錯誤模式（前10個）:
  predicted=right, gt=left: 12 次
  predicted=2, gt=1: 5 次
  predicted=3, gt=2: 3 次
  predicted=1, gt=3: 2 次
  predicted=2, gt=4: 2 次
  predicted=3, gt=4: 1 次
  predicted=9, gt=8: 1 次
  predicted=left, gt=right: 1 次
  predicted=0.49, gt=none: 1 次
  predicted=0, gt=1: 1 次

詳細錯誤類型:
  方向判斷錯誤: 13 個
    Left→Right 錯誤: 1 次
    Right→Left 錯誤: 12 次
    → 建議: 可能存在右左判斷偏差，檢查 is_left/is_right 工具函數
  計數錯誤: 15 個
    平均誤差: -0.07
    平均絕對誤差: 1.27
    差1個: 11 次
    差2個: 4 次
  距離估計錯誤: 7 個
    平均相對誤差: 19.6%
  選擇題錯誤: 4 個
    → 建議: 檢查物體識別和空間推理邏輯

錯誤案例已保存到: ../output/val_errors.json
詳細錯誤報告已保存到: ../output/val_errors_detailed.json
結果已保存到: ../output/val_eval_results.json


---
# hybrid (dist: add_shortcut_best.pth, closest: log_mse_best_model.pth)

```bash
python train_eval.py --dataset val --limit 630 --verbose --llm_type vllm --api_base http://localhost:8040/v1 --model_name Qwen/Qwen3-4B-Instruct-2507 --categories distance
```
準確率報告 (val set):
嚴格模式 (±0.1): 74.00% (111/150)
10% 容錯模式: 94.67% (142/150)
20% 容錯模式: 95.33% (143/150)
預設使用: 10% 容錯模式

問題分類準確率: 100.00% (150/150)

分類準確率（按類別）:
  distance: 100.00% (150/150)

錯誤分析 (val set):
總錯誤數: 8

按類別分布:
  distance: 8 個錯誤

常見錯誤模式（前10個）:
  predicted=0.58, gt=0: 2 次
  predicted=0.53, gt=0: 1 次
  predicted=3.83, gt=14.64: 1 次
  predicted=0.56, gt=0: 1 次
  predicted=0.47, gt=0: 1 次
  predicted=0.11, gt=0: 1 次
  predicted=1.21, gt=1.36: 1 次

詳細錯誤類型:
  距離估計錯誤: 8 個
    平均相對誤差: 42.4%
    → 建議: 距離估計模型可能需要重新訓練或調整

錯誤案例已保存到: ../output/val_errors.json
詳細錯誤報告已保存到: ../output/val_errors_detailed.json
結果已保存到: ../output/val_eval_results.json

---
# distance: log_mse_best_model.pth, closest: log_mse_best_model.pth

準確率報告 (val set):
嚴格模式 (±0.1): 36.67% (55/150)
10% 容錯模式: 85.33% (128/150)
20% 容錯模式: 90.67% (136/150)
預設使用: 10% 容錯模式

問題分類準確率: 100.00% (150/150)

分類準確率（按類別）:
  distance: 100.00% (150/150)

錯誤分析 (val set):
總錯誤數: 22

按類別分布:
  distance: 22 個錯誤

常見錯誤模式（前10個）:
  predicted=0.49, gt=0: 1 次
  predicted=1.47, gt=1.18: 1 次
  predicted=2.22, gt=2.62: 1 次
  predicted=3.62, gt=14.64: 1 次
  predicted=1.63, gt=2.07: 1 次
  predicted=0.59, gt=0: 1 次
  predicted=2.46, gt=3.07: 1 次
  predicted=1.39, gt=1.19: 1 次
  predicted=3.63, gt=4.35: 1 次
  predicted=1.02, gt=0.83: 1 次

詳細錯誤類型:
  距離估計錯誤: 22 個
    平均相對誤差: 23.0%
    → 建議: 距離估計模型可能需要重新訓練或調整

錯誤案例已保存到: ../output/val_errors.json
詳細錯誤報告已保存到: ../output/val_errors_detailed.json
結果已保存到: ../output/val_eval_results.json

---
# distance: add_shortcut_best.pth, closest: closest_pred
準確率報告 (val set):
嚴格模式 (±0.1): 74.67% (112/150)
10% 容錯模式: 95.33% (143/150)
20% 容錯模式: 96.00% (144/150)
預設使用: 10% 容錯模式

問題分類準確率: 100.00% (150/150)

分類準確率（按類別）:
  distance: 100.00% (150/150)

錯誤分析 (val set):
總錯誤數: 7

按類別分布:
  distance: 7 個錯誤

常見錯誤模式（前10個）:
  predicted=0.58, gt=0: 2 次
  predicted=0.53, gt=0: 1 次
  predicted=0.56, gt=0: 1 次
  predicted=0.47, gt=0: 1 次
  predicted=0.11, gt=0: 1 次
  predicted=1.21, gt=1.36: 1 次

詳細錯誤類型:
  距離估計錯誤: 7 個
    平均相對誤差: 11.0%

錯誤案例已保存到: ../output/val_errors.json
詳細錯誤報告已保存到: ../output/val_errors_detailed.json
結果已保存到: ../output/val_eval_results.json

---
python train_eval.py --dataset val --limit 300 --verbose --llm_type vllm --api_base http://localhost:8040/v1 --model_name Qwen/Qwen3-4B-Instruct-2507
```bash
```
準確率報告 (val set):
嚴格模式 (±0.1): 82.67% (248/300)
10% 容錯模式: 86.33% (259/300)
20% 容錯模式: 87.33% (262/300)
預設使用: 10% 容錯模式

問題分類準確率: 100.00% (300/300)

分類準確率（按類別）:
  left_right: 100.00% (84/84)
  count: 100.00% (74/74)
  distance: 100.00% (63/63)
  mcq: 100.00% (79/79)

錯誤分析 (val set):
總錯誤數: 41

按類別分布:
  mcq: 23 個錯誤
  count: 10 個錯誤
  distance: 3 個錯誤
  left_right: 5 個錯誤

常見錯誤模式（前10個）:
  predicted=2, gt=1: 5 次
  predicted=right, gt=left: 5 次
  predicted=2, gt=5: 3 次
  predicted=3, gt=2: 3 次
  predicted=4, gt=3: 3 次
  predicted=3, gt=4: 2 次
  predicted=2, gt=0: 2 次
  predicted=5, gt=3: 1 次
  predicted=9.21, gt=11.0: 1 次
  predicted=9, gt=3: 1 次

詳細錯誤類型:
  方向判斷錯誤: 5 個
    Left→Right 錯誤: 0 次
    Right→Left 錯誤: 5 次
    → 建議: 可能存在右左判斷偏差，檢查 is_left/is_right 工具函數
  計數錯誤: 10 個
    平均誤差: 0.80
    平均絕對誤差: 1.00
    差1個: 10 次
    差2個: 0 次
    → 建議: 系統傾向高估數量，可能計算了多餘的物體
  距離估計錯誤: 3 個
    平均相對誤差: 16.3%
  選擇題錯誤: 23 個
    → 建議: 檢查物體識別和空間推理邏輯

錯誤案例已保存到: ../output/val_errors.json
詳細錯誤報告已保存到: ../output/val_errors_detailed.json
結果已保存到: ../output/val_eval_results.json

---
modify in tools.py

準確率報告 (val set):
嚴格模式 (±0.1): 76.67% (115/150)
10% 容錯模式: 97.33% (146/150)
20% 容錯模式: 98.00% (147/150)
預設使用: 10% 容錯模式

問題分類準確率: 100.00% (150/150)

分類準確率（按類別）:
  distance: 100.00% (150/150)

錯誤分析 (val set):
總錯誤數: 4

按類別分布:
  distance: 4 個錯誤

常見錯誤模式（前10個）:
  predicted=0.58, gt=0: 2 次
  predicted=0.0, gt=2.62: 1 次
  predicted=1.21, gt=1.36: 1 次

詳細錯誤類型:
  距離估計錯誤: 4 個
    平均相對誤差: 55.5%
    → 建議: 距離估計模型可能需要重新訓練或調整

錯誤案例已保存到: ../output/val_errors.json
詳細錯誤報告已保存到: ../output/val_errors_detailed.json
結果已保存到: ../output/val_eval_results.json

---
### generating data for closest data
```bash
python generate_closest_data.py     --train_json ../../DLCV_Final1/train.json     --image_dir ../../DLCV_Final1/train/images     --output_json ./closest_data/train.json     --max_images 5000     --synthetic_samples 5
```

Loading data from ../../DLCV_Final1/train.json...
Loaded 217805 samples

### training closest model
```bash
python train.py \
    --data_root ./closest_data \
    --train_json train.json \
    --output_dir ./checkpoints \
    --backbone resnet50 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --val_split 0.1
```

---
### tighter threshold for inclusive (inside_thres 0.5-> 0.6, add IoU minimum check)

準確率報告 (val set):
嚴格模式 (±0.1): 88.67% (266/300)
10% 容錯模式: 92.67% (278/300)
20% 容錯模式: 93.00% (279/300)
預設使用: 10% 容錯模式

問題分類準確率: 100.00% (300/300)

分類準確率（按類別）:
  left_right: 100.00% (84/84)
  count: 100.00% (74/74)
  distance: 100.00% (63/63)
  mcq: 100.00% (79/79)

錯誤分析 (val set):
總錯誤數: 22

按類別分布:
  count: 12 個錯誤
  left_right: 4 個錯誤
  mcq: 4 個錯誤
  distance: 2 個錯誤

常見錯誤模式（前10個）:
  predicted=2, gt=1: 4 次
  predicted=3, gt=4: 4 次
  predicted=right, gt=left: 4 次
  predicted=3, gt=2: 2 次
  predicted=9, gt=8: 1 次
  predicted=0.53, gt=0: 1 次
  predicted=1, gt=0: 1 次
  predicted=2, gt=0: 1 次
  predicted=2, gt=4: 1 次
  predicted=0.58, gt=0: 1 次

詳細錯誤類型:
  方向判斷錯誤: 4 個
    Left→Right 錯誤: 0 次
    Right→Left 錯誤: 4 次
    → 建議: 可能存在右左判斷偏差，檢查 is_left/is_right 工具函數
  計數錯誤: 12 個
    平均誤差: 0.00
    平均絕對誤差: 1.00
    差1個: 12 次
    差2個: 0 次
  距離估計錯誤: 2 個
  選擇題錯誤: 4 個
    → 建議: 檢查物體識別和空間推理邏輯

錯誤案例已保存到: ../output/val_errors.json
詳細錯誤報告已保存到: ../output/val_errors_detailed.json
結果已保存到: ../output/val_eval_results.json