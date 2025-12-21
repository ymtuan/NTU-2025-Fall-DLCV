import json
import os
import argparse
import numpy as np
import random
import re
from tqdm import tqdm
from openai import OpenAI
import pycocotools.mask as mask_utils
from collections import defaultdict

# ==========================================
# 1. LLM Extractor Class (vLLM Integrated)
# ==========================================

class LocalLLMExtractor:
    def __init__(self, args):
        self.model_name = args.model_name
        self.client_type = args.llm_type
        self.api_base = args.api_base
        self.api_key = args.api_key
        
        # 檢測是否為 Vision Model (雖然此腳本只用文字能力，但保留參數邏輯)
        model_name_lower = self.model_name.lower()
        self.is_vision = any(keyword in model_name_lower for keyword in ['vl', 'vision', 'visual'])
        
        print(f"[{self.client_type.upper()}] Initializing client...")
        print(f"Target: {self.api_base} | Model: {self.model_name} | Vision: {self.is_vision}")

        # 初始化 OpenAI Client (用於連接 vLLM)
        if self.client_type == 'vllm' or self.client_type == 'openai':
            self.client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
        else:
            raise ValueError(f"Unsupported llm_type: {self.client_type}")

        # System Prompt: 強制要求 JSON 格式
        self.system_prompt = (
            "You are a semantic parser for warehouse data. "
            "Your task is to analyze the description and extract relationships where an Object is strictly INSIDE, ON, or CONTAINED BY a Region. "
            "Input text contains references like '[Region X]'. "
            "Output valid JSON only. Format: {\"pairs\": [{\"object_id\": int, \"region_id\": int}]}. "
            "Rules:\n"
            "1. Only extract explicit containment (e.g., 'inside', 'in', 'on', 'stored in').\n"
            "2. Ignore 'closest to', 'next to', or 'left/right of'.\n"
            "3. If no inclusion relationship exists, return {\"pairs\": []}."
        )

    def extract_positives(self, text):
        """
        Call LLM to parse text and return list of tuple (obj_id, region_id)
        """
        if not text:
            return []

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Description: {text}"}
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # 低溫度以獲得穩定輸出
                max_tokens=256,
                # 如果模型支援 json mode (如 Llama3)，建議開啟:
                # response_format={"type": "json_object"}, 
            )
            
            content = response.choices[0].message.content
            return self._parse_json_response(content)
            
        except Exception as e:
            print(f"\n[Error] LLM Call Failed: {e}")
            return []

    def _parse_json_response(self, json_str):
        """Clean and parse LLM output"""
        try:
            # 清理 Markdown Code Block
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            # 嘗試解析
            data = json.loads(json_str)
            
            # 處理不同可能的 JSON 結構
            if isinstance(data, list):
                pairs = data
            elif isinstance(data, dict):
                pairs = data.get('pairs', [])
            else:
                return []

            results = []
            for p in pairs:
                # 確保 ID 存在且為整數
                if 'object_id' in p and 'region_id' in p:
                    results.append((int(p['object_id']), int(p['region_id'])))
            return results

        except json.JSONDecodeError:
            # 如果 LLM 吐出的不是 JSON，嘗試用 Regex 補救
            # pattern = r'object_id":\s*(\d+).*?region_id":\s*(\d+)'
            # matches = re.findall(pattern, json_str)
            # return [(int(o), int(r)) for o, r in matches]
            return []

# ==========================================
# 2. Geometry Utils
# ==========================================

def decode_rle(rle_dict):
    """Decode RLE mask to numpy array"""
    if isinstance(rle_dict, list): 
        return mask_utils.decode(rle_dict).astype(np.uint8)
    return mask_utils.decode(rle_dict).astype(np.uint8)

def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union"""
    intersection = np.sum(mask1 & mask2) # Bitwise AND for binary masks is faster
    union = np.sum(mask1 | mask2)
    if union == 0: return 0.0
    return intersection / union

def generate_hard_negatives(masks, positive_pairs, num_needed, hard_ratio=0.7):
    """
    Generate negative samples with a mix of Hard Negatives (High IoU) and Easy Negatives.
    
    Args:
        masks: List of decoded binary masks
        positive_pairs: List of (obj_id, container_id) tuples that are TRUE positives
        num_needed: Total number of negative samples needed
        hard_ratio: Percentage of samples that should be "Hard" (High IoU)
    
    Returns:
        List of (idx1, idx2) tuples representing negative pairs
    """
    num_objs = len(masks)
    positive_set = set(positive_pairs)
    
    # 1. 找出所有可能的 Negative Candidates (排除自己和已知的 Positives)
    candidates = []
    
    # 對於物件數量不多的圖 (N < 100)，雙重迴圈計算所有 IoU 是很快的
    for i in range(num_objs):
        for j in range(num_objs):
            if i == j: 
                continue
            
            # 檢查是否為 Positive (雙向檢查，視為互斥)
            if (i, j) in positive_set or (j, i) in positive_set:
                continue
            
            # 計算 IoU
            iou = calculate_iou(masks[i], masks[j])
            candidates.append({'idx1': i, 'idx2': j, 'iou': iou})
    
    # 2. 根據 IoU 排序 (由大到小) -> IoU 越大越 Hard
    # 我們希望模型學會：即使 IoU 很大 (0.6, 0.7)，只要它不在 positive list 裡，就是 Negative
    candidates.sort(key=lambda x: x['iou'], reverse=True)
    
    selected_negatives = []
    
    # 3. 選取 Hard Negatives (高 IoU)
    num_hard = int(num_needed * hard_ratio)
    
    # 從候選列表中取出前 N 個 (IoU 最高的)
    # 注意：要過濾掉 IoU 太接近 1.0 的 (例如 > 0.95)，那可能是標註漏掉的 Positive，避免汙染
    hard_candidates = [c for c in candidates if c['iou'] < 0.95]
    
    # 取出 Hard samples
    for item in hard_candidates[:num_hard]:
        selected_negatives.append((item['idx1'], item['idx2']))
    
    # 4. 選取 Easy Negatives (隨機補足剩下的)
    # 從剩下的候選者中隨機選，或者選 IoU=0 的
    num_easy = num_needed - len(selected_negatives)
    if num_easy > 0:
        remaining_candidates = hard_candidates[num_hard:]
        if remaining_candidates:
            # 隨機打亂剩下的，選出 Easy samples
            random.shuffle(remaining_candidates)
            for item in remaining_candidates[:num_easy]:
                selected_negatives.append((item['idx1'], item['idx2']))
        else:
            # 如果候選者不夠 (圖很空)，就隨機生成 (雖然可能重複，但機率低)
            # 這裡簡單處理：如果不夠就只回傳目前的
            pass

    return selected_negatives

# ==========================================
# 3. Main Data Processing Logic
# ==========================================

def create_dataset(args):
    # 1. 初始化 LLM Client
    llm = LocalLLMExtractor(args)
    
    # 2. 讀取原始資料
    print(f"Loading raw data from: {args.input_json}")
    with open(args.input_json, 'r') as f:
        raw_data = json.load(f)
        
    if args.max_samples:
        raw_data = raw_data[:args.max_samples]
        print(f"Debug Mode: Processing only {args.max_samples} images.")

    # 3. 讀取已處理的資料（如果存在）
    dataset = []
    stats = defaultdict(int)
    processed_images = set()
    start_idx = 0
    
    if os.path.exists(args.output_json):
        print(f"Found existing dataset at: {args.output_json}")
        with open(args.output_json, 'r') as f:
            dataset = json.load(f)
        
        # 統計已處理的圖片
        for item in dataset:
            processed_images.add(item['image'])
            stats['positives' if item['inside'] == 1 else 'negatives'] += 1
        
        print(f"Loaded {len(dataset)} existing samples from {len(processed_images)} images")
        print(f"  - Positives: {stats['positives']}, Negatives: {stats['negatives']}")

    print("Start processing...")
    checkpoint_interval = 5000  # 每 5000 筆存一次
    samples_since_last_save = 0
    
    for idx, item in enumerate(tqdm(raw_data)):
        image_path = item['image']
        
        # 跳過已處理的圖片
        if image_path in processed_images:
            continue
        
        rle_masks = item['rle']
        
        # 3. 獲取描述文本
        text_context = ""
        # 優先使用 freeform_answer
        if 'freeform_answer' in item:
            text_context = item['freeform_answer']
        # 其次嘗試從對話中提取 gpt 的回答
        elif 'conversations' in item:
            for turn in item['conversations']:
                if turn['from'] == 'gpt':
                    text_context += turn['value'] + " "
        
        if not text_context:
            continue

        # 4. LLM 提取 Positive Pairs
        pos_pairs = llm.extract_positives(text_context)
        
        # 驗證 ID 是否有效
        valid_pos_pairs = []
        num_masks = len(rle_masks)
        for obj_id, reg_id in pos_pairs:
            if obj_id < num_masks and reg_id < num_masks and obj_id != reg_id:
                valid_pos_pairs.append((obj_id, reg_id))
        
        # 去重
        valid_pos_pairs = list(set(valid_pos_pairs))
        
        # 5. 如果有正樣本，生成對應數量的負樣本
        if valid_pos_pairs:
            # 為了效率，只在此時解碼這張圖的所有 Mask
            decoded_masks = [decode_rle(r) for r in rle_masks]
            
            # 生成負樣本 (使用新的 Hard Negative 邏輯)
            neg_pairs = generate_hard_negatives(
                decoded_masks, 
                valid_pos_pairs, 
                num_needed=len(valid_pos_pairs), # 1:1 平衡
                hard_ratio=args.hard_ratio # 70% 的負樣本來自高重疊區域
            )
            
            # 6. 寫入資料集
            # Positive Samples
            for obj_id, reg_id in valid_pos_pairs:
                dataset.append({
                    'image': image_path,              # 符合 data_loader.py
                    'obj_rle': rle_masks[obj_id],     # 符合 data_loader.py
                    'buffer_rle': rle_masks[reg_id],  # 符合 data_loader.py
                    'inside': 1,                      # 符合 data_loader.py
                    'freeform_answer': text_context,  # 保留原始文本對照
                    'meta': {'obj_id': obj_id, 'region_id': reg_id, 'source': 'llm'}
                })
                stats['positives'] += 1
                samples_since_last_save += 1
                
            # Negative Samples
            for obj_id, reg_id in neg_pairs:
                dataset.append({
                    'image': image_path,              # 符合 data_loader.py
                    'obj_rle': rle_masks[obj_id],     # 符合 data_loader.py
                    'buffer_rle': rle_masks[reg_id],  # 符合 data_loader.py
                    'inside': 0,                      # 符合 data_loader.py
                    'freeform_answer': text_context,  # 保留原始文本對照
                    'meta': {'obj_id': obj_id, 'region_id': reg_id, 'source': 'iou_neg'}
                })
                stats['negatives'] += 1
                samples_since_last_save += 1
            
            # 記錄已處理的圖片
            processed_images.add(image_path)
            
            # 7. 定期存檔 (每 5000 筆)
            if samples_since_last_save >= checkpoint_interval:
                print(f"\n[Checkpoint] Saving {len(dataset)} samples to {args.output_json}...")
                os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else '.', exist_ok=True)
                with open(args.output_json, 'w') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                print(f"[Checkpoint] Saved! (Positives: {stats['positives']}, Negatives: {stats['negatives']})")
                samples_since_last_save = 0

    # 8. 最終存檔與統計
    print("\n" + "="*40)
    print("Dataset Generation Complete")
    print(f"Total Images Processed: {len(processed_images)}")
    print(f"Total Samples: {len(dataset)}")
    print(f"  - Positives (Inside): {stats['positives']}")
    print(f"  - Negatives (Outside): {stats['negatives']}")
    print("="*40)

    # Final Save
    print(f"Saving final dataset to {args.output_json}...")
    os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else '.', exist_ok=True)
    
    # Shuffle before final save
    random.shuffle(dataset)
    
    with open(args.output_json, 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Final dataset saved to: {args.output_json}")

# ==========================================
# 4. Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Inclusion Dataset with vLLM')
    
    # Path Arguments
    parser.add_argument('--input_json', type=str, default='/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/data/train.json')
    parser.add_argument('--output_json', type=str, default='./data/inclusion_train.json')
    parser.add_argument('--max_samples', type=int, default=None, help='Debug: limit number of images')
    
    # LLM Connection Arguments
    parser.add_argument('--llm_type', type=str, default='vllm', choices=['vllm', 'openai'])
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--api_base', type=str, default='http://localhost:8040/v1')
    parser.add_argument('--api_key', type=str, default='EMPTY')
    
    # Algorithm Arguments
    parser.add_argument('--neg_iou_thresh', type=float, default=0.2, help='[Deprecated] Max IoU for negative samples (no longer used)')
    parser.add_argument('--hard_ratio', type=float, default=0.7, help='Ratio of hard negatives (high IoU) in negative samples (default: 0.7)')

    args = parser.parse_args()
    
    create_dataset(args)