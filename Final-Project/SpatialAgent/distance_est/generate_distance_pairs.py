import json
import os
import argparse
import re
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI

# ==========================================
# 1. LLM Extractor Class
# ==========================================

class DistanceLLMExtractor:
    def __init__(self, args):
        self.model_name = args.model_name
        self.client_type = args.llm_type
        self.api_base = args.api_base
        self.api_key = args.api_key
        
        print(f"[{self.client_type.upper()}] Initializing client...")
        print(f"Target: {self.api_base} | Model: {self.model_name}")

        # 初始化 OpenAI Client (用於連接 vLLM)
        if self.client_type == 'vllm' or self.client_type == 'openai':
            self.client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
        else:
            raise ValueError(f"Unsupported llm_type: {self.client_type}")

        # System Prompt: 強制要求提取距離資訊
        self.system_prompt = (
            "You are a distance information extractor for warehouse data. "
            "Your task is to analyze the description and extract distance measurements between regions. "
            "Input text contains references like '[Region X]' and distance values in meters. "
            "Output valid JSON only. Format: {\"pairs\": [{\"region_id1\": int, \"region_id2\": int, \"distance\": float}]}. "
            "Rules:\n"
            "1. Only extract explicit distance measurements (e.g., 'X meters', 'distance of X meters').\n"
            "2. Extract the two region IDs involved in the measurement.\n"
            "3. Extract the distance value as a float (in meters).\n"
            "4. If no distance relationship exists, return {\"pairs\": []}.\n"
            "5. Ignore comparisons without explicit measurements (e.g., 'closest to', 'nearest')."
        )

    def extract_distances(self, text):
        """
        Call LLM to parse text and return list of tuple (region_id1, region_id2, distance)
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
                max_tokens=512,
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
                # 確保所有欄位存在且為正確型別
                if 'region_id1' in p and 'region_id2' in p and 'distance' in p:
                    try:
                        region1 = int(p['region_id1'])
                        region2 = int(p['region_id2'])
                        distance = float(p['distance'])
                        results.append((region1, region2, distance))
                    except (ValueError, TypeError):
                        continue
            return results

        except json.JSONDecodeError:
            # JSON 解析失敗，返回空列表
            return []


def create_distance_dataset(args):
    """
    從原始資料中提取距離配對，生成訓練資料集
    """
    # 1. 初始化 LLM Client
    llm = DistanceLLMExtractor(args)
    
    # 2. 讀取原始資料 - 使用更穩健的方式
    print(f"Loading data from: {args.input_json}")
    raw_data = []
    
    # 先嘗試正常載入
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        print(f"✓ Successfully loaded {len(raw_data)} items")
    except json.JSONDecodeError as e:
        print(f"✗ JSON decode error at line {e.lineno}, col {e.colno}")
        print(f"  Attempting to use streaming parser...")
        
        # 使用逐行解析（假設每個物件佔一行或可以用 ijson）
        try:
            import ijson
            with open(args.input_json, 'rb') as f:
                raw_data = list(ijson.items(f, 'item'))
            print(f"✓ Streaming parser loaded {len(raw_data)} items")
        except ImportError:
            print("  ijson not installed, trying alternative method...")
            # 讀取到錯誤位置之前的資料
            with open(args.input_json, 'r', encoding='utf-8') as f:
                content = f.read(e.pos)
                # 找最後一個完整的物件
                last_bracket = content.rfind('}')
                if last_bracket != -1:
                    # 嘗試解析到最後一個完整物件
                    truncated = content[:last_bracket+1] + ']'
                    raw_data = json.loads(truncated)
                    print(f"✓ Recovered {len(raw_data)} items (truncated at error)")
        except Exception as parse_error:
            print(f"  Failed: {parse_error}")
            print(f"  Please fix the JSON file manually")
            return
    
    if not raw_data:
        print("Error: No data loaded!")
        return
    
    if args.max_samples:
        raw_data = raw_data[:args.max_samples]
        print(f"Debug Mode: Processing only {args.max_samples} samples.")
    
    # 3. 讀取已處理的資料（如果存在）
    dataset = []
    processed_ids = set()
    
    if os.path.exists(args.output_json):
        print(f"Found existing dataset at: {args.output_json}")
        with open(args.output_json, 'r') as f:
            dataset = json.load(f)
        
        # 統計已處理的項目
        for item in dataset:
            processed_ids.add(item['id'])
        
        print(f"Loaded {len(dataset)} existing samples")
    
    stats = {
        'total_items': len(raw_data),
        'items_with_distance': 0,
        'total_pairs': 0,
        'skipped_invalid': 0,
        'skipped_processed': 0
    }
    
    checkpoint_interval = 100  # 每 100 筆存一次
    samples_since_last_save = 0
    
    print("Extracting distance pairs with LLM...")
    for item in tqdm(raw_data):
        item_id = item.get('id', '')
        
        # 跳過已處理的項目
        if item_id in processed_ids:
            stats['skipped_processed'] += 1
            continue
        
        image_path = item.get('image', '')
        rle_masks = item.get('rle', [])
        num_masks = len(rle_masks)
        
        # 獲取答案文本
        text_context = ""
        
        # 優先使用 freeform_answer
        if 'freeform_answer' in item:
            text_context = item['freeform_answer']
        # 其次從 conversations 中提取
        elif 'conversations' in item:
            for turn in item['conversations']:
                if turn.get('from') == 'gpt' or turn.get('role') == 'assistant':
                    value = turn.get('value', '') or turn.get('content', '')
                    text_context += value + " "
        
        if not text_context:
            continue
        
        # 使用 LLM 提取距離配對
        pairs = llm.extract_distances(text_context)
        
        if not pairs:
            continue
        
        stats['items_with_distance'] += 1
        
        # 驗證並添加到資料集
        for region1, region2, distance in pairs:
            # 驗證 region ID 是否有效
            if region1 >= num_masks or region2 >= num_masks:
                stats['skipped_invalid'] += 1
                if args.verbose:
                    print(f"\nWarning: Invalid region IDs ({region1}, {region2}) for image {image_path} with {num_masks} masks")
                continue
            
            # 同一個 region 不計算距離
            if region1 == region2:
                stats['skipped_invalid'] += 1
                continue
            
            # 獲取問題文本（用於參考）
            question = ""
            if 'conversations' in item:
                for turn in item['conversations']:
                    if turn.get('from') == 'human' or turn.get('role') == 'user':
                        question = turn.get('value', '') or turn.get('content', '')
                        break
            
            # 添加到資料集（格式與 val_dist_est.json 一致）
            dataset.append({
                'id': f"{item_id}_{region1}_{region2}" if item_id else f"{image_path}_{region1}_{region2}",
                'image': image_path,
                'category': 'distance',
                'conversations': [
                    {
                        'from': 'human',
                        'value': question
                    },
                    {
                        'from': 'gpt',
                        'value': text_context.strip()
                    }
                ],
                'rle': [rle_masks[region1], rle_masks[region2]],
                'mask_ids': [region1, region2],
                'distance': distance
            })
            stats['total_pairs'] += 1
            samples_since_last_save += 1
        
        # 記錄已處理的 ID
        processed_ids.add(item_id)
        
        # 定期存檔
        if samples_since_last_save >= checkpoint_interval:
            print(f"\n[Checkpoint] Saving {len(dataset)} samples to {args.output_json}...")
            output_dir = os.path.dirname(args.output_json)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output_json, 'w') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"[Checkpoint] Saved! (Total pairs: {stats['total_pairs']})")
            samples_since_last_save = 0
    
    # 統計報告
    print("\n" + "="*60)
    print("Distance Dataset Generation Complete")
    print("="*60)
    print(f"Total input items:        {stats['total_items']}")
    print(f"Items with distance info: {stats['items_with_distance']}")
    print(f"Total distance pairs:     {stats['total_pairs']}")
    print(f"Skipped (invalid):        {stats['skipped_invalid']}")
    print(f"Skipped (processed):      {stats['skipped_processed']}")
    print("="*60)
    
    # 最終保存資料集
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving final dataset to: {args.output_json}")
    with open(args.output_json, 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved successfully!")
    
    # 顯示一些範例
    if dataset and args.verbose:
        print("\n" + "="*60)
        print("Sample entries:")
        print("="*60)
        for i, sample in enumerate(dataset[:3]):
            print(f"\nSample {i+1}:")
            print(f"  ID: {sample['id']}")
            print(f"  Image: {sample['image']}")
            print(f"  Mask IDs: {sample['mask_ids']}")
            print(f"  Distance: {sample['distance']} meters")
            print(f"  Answer: {sample['conversations'][1]['value'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Distance Pairs Dataset from LLM Answers')
    
    # Path Arguments
    parser.add_argument('--input_json', type=str, required=True, 
                       help='Path to input JSON (train.json or val.json)')
    parser.add_argument('--output_json', type=str, required=True,
                       help='Path to output distance dataset JSON')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Debug: limit number of samples to process')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    
    # LLM Connection Arguments
    parser.add_argument('--llm_type', type=str, default='vllm', choices=['vllm', 'openai'])
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--api_base', type=str, default='http://localhost:8000/v1')
    parser.add_argument('--api_key', type=str, default='EMPTY')
    
    args = parser.parse_args()
    
    create_distance_dataset(args)
