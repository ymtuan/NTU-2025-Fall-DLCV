# Train Set 評估指南

## 功能說明

`train_eval.py` 用於在 train set 上評估模型表現，並分析錯誤案例。

## 使用方法

### 1. 使用 Gemini (Vertex AI)

```bash
cd agent
python train_eval.py \
    --project_id gen-lang-client-0647114394 \
    --llm_type gemini \
    --limit 1000 \
    --output_dir ../output
```

### 2. 使用 vLLM (本地模型)

#### 步驟 1: 啟動 vLLM 服務器

```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.7 \
    --port 8040
```

#### 步驟 2: 運行評估

```bash
cd agent
python train_eval.py \
    --llm_type vllm \
    --api_base http://localhost:8040/v1 \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --limit 1000 \
    --output_dir ../output
```

### 3. 參數說明

- `--llm_type`: LLM 類型 (`gemini`, `openai`, `vllm`)
- `--project_id`: Google Cloud Project ID (僅 Gemini 需要)
- `--api_base`: vLLM API 基礎 URL (預設: `http://localhost:8040/v1`)
- `--api_key`: API 密鑰 (vLLM 可使用任意值，預設: `dummy`)
- `--model_name`: 模型名稱 (預設: `Qwen/Qwen3-4B-Instruct-2507`)
- `--temperature`: 溫度參數 (預設: 0.2)
- `--max_tokens`: 最大 token 數 (預設: 2048)
- `--limit`: 處理的樣本數量 (預設: 1000)
- `--output_dir`: 輸出目錄 (預設: `../output`)

## 輸出文件

評估完成後會生成：

1. `train_eval_results.json`: 所有結果
   ```json
   [
     {
       "id": "...",
       "predicted": "pallet_1",
       "ground_truth": "pallet_1",
       "correct": true,
       "category": "mcq",
       "conversation": [...]
     },
     ...
   ]
   ```

2. `train_errors.json`: 錯誤案例
   - 只包含 `correct: false` 的項目

## 錯誤分析

腳本會自動分析：
- 按類別 (category) 的錯誤分布
- 常見錯誤模式
- 錯誤案例詳情

## 注意事項

1. **圖片文件**: 確保 `data/train/images/` 已解壓縮
2. **vLLM 服務器**: 確保 vLLM 服務器正在運行
3. **依賴套件**: 使用 vLLM 需要安裝 `openai`:
   ```bash
   pip install openai
   ```

## 範例輸出

```
載入 train set...
找到 1000 個有圖片的項目（限制 1000 筆）
使用 vLLM/OpenAI API: http://localhost:8040/v1
模型: Qwen/Qwen3-4B-Instruct-2507
Processing: 100%|████████| 1000/1000 [15:23<00:00, 1.08it/s]

準確率: 85.30% (853/1000)

錯誤分析:
總錯誤數: 147

按類別分布:
  mcq: 45 個錯誤
  distance: 38 個錯誤
  left_right: 42 個錯誤
  count: 22 個錯誤

常見錯誤模式（前10個）:
  predicted=pallet_0, gt=pallet_1: 12 次
  predicted=2.5, gt=2.3: 8 次
  ...

結果已保存到: ../output/train_eval_results.json
錯誤案例已保存到: ../output/train_errors.json
```

