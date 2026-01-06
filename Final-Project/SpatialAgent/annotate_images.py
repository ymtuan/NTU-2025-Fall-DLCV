#!/usr/bin/env python3
"""
根據 val.json 的資訊在圖像上標註物件代號（如 buffer_0, pallet_1）
用於調試和視覺化
"""

import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse

# 添加 agent 目錄到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))
from mask import parse_masks_from_conversation, Mask
from llm_client import create_llm_client


def get_centroid(mask_array):
    """計算 mask 的質心"""
    y_indices, x_indices = np.where(mask_array > 0)
    if len(x_indices) == 0:
        return None
    return (int(np.mean(x_indices)), int(np.mean(y_indices)))


def get_mask_area(mask_array):
    """計算 mask 的面積（像素數）"""
    return np.sum(mask_array > 0)


def calculate_dynamic_font_size(mask_area, img_area, base_font_size=30, min_size=15, max_size=40):
    """根據 mask 面積動態計算字體大小"""
    # 計算 mask 佔圖像的比例
    area_ratio = mask_area / img_area if img_area > 0 else 0
    
    # 根據比例調整字體大小
    if area_ratio > 0.1:  # 大物體
        font_size = max_size
    elif area_ratio > 0.05:  # 中等物體
        font_size = base_font_size
    elif area_ratio > 0.01:  # 小物體
        font_size = max(min_size, base_font_size * 0.7)
    else:  # 很小物體
        font_size = min_size
    
    return int(font_size)


def annotate_image(image_path, masks_dict, output_path, font_size=30, use_region_id=False):
    """
    在圖像上繪製 masks 並標註物件名稱
    
    Args:
        image_path: 原始圖像路徑
        masks_dict: {mask_name: Mask} 字典
        output_path: 輸出圖像路徑
        font_size: 字體大小
        use_region_id: 已棄用，現在總是使用 mask_name 格式（例如 "buffer_0", "pallet_1"）
    """
    # 載入圖像
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # 創建可繪製的圖像副本
    annotated_img = img.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # 定義不同類別的顏色（優化後：fill 低透明度，outline 高對比度不透明）
    class_colors = {
        'buffer': {'outline': (255, 0, 0, 255), 'fill': (255, 0, 0, 40)},      # 紅色
        'pallet': {'outline': (0, 255, 0, 255), 'fill': (0, 255, 0, 40)},      # 綠色
        'transporter': {'outline': (0, 0, 255, 255), 'fill': (0, 0, 255, 40)}, # 藍色
        'shelf': {'outline': (255, 165, 0, 255), 'fill': (255, 165, 0, 40)},   # 橙色
        'object': {'outline': (128, 128, 128, 255), 'fill': (128, 128, 128, 40)} # 灰色
    }
    
    # 計算圖像總面積（用於動態字體大小）
    img_area = img.width * img.height
    
    # 創建一個統一的 mask overlay 圖層（用於繪製所有 masks）
    mask_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_overlay)
    
    # 為每個 mask 繪製和標註
    for mask_name, mask_obj in masks_dict.items():
        # 解碼 mask
        mask_array = mask_obj.decode_mask()
        
        # 獲取類別顏色樣式
        class_name = mask_obj.object_class.lower()
        style = class_colors.get(class_name, class_colors['object'])
        
        # 找到 mask 的邊界
        y_indices, x_indices = np.where(mask_array > 0)
        if len(x_indices) > 0:
            # 計算 mask 面積（用於動態字體大小）
            mask_area = get_mask_area(mask_array)
            dynamic_font_size = calculate_dynamic_font_size(mask_area, img_area, font_size)
            
            # 載入動態字體
            try:
                dynamic_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", dynamic_font_size)
            except:
                try:
                    dynamic_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", dynamic_font_size)
                except:
                    dynamic_font = ImageFont.load_default()
            
            # 計算邊界框
            x_min, x_max = int(x_indices.min()), int(x_indices.max())
            y_min, y_max = int(y_indices.min()), int(y_indices.max())
            outline_color = style['outline'][:3]  # RGB only
            
            # 1. 繪製精確的 Mask 填充（低透明度）
            # 創建單獨的 mask 圖層
            single_mask_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            single_mask_array = np.array(single_mask_layer)
            
            # 填充 mask 區域（使用 numpy 操作，效率更高）
            fill_color = style['fill']
            single_mask_array[y_indices, x_indices] = fill_color
            
            # 轉回 PIL Image
            single_mask_layer = Image.fromarray(single_mask_array, mode='RGBA')
            
            # 2. 在單個 mask 圖層上繪製邊框（確保邊框在填充之上）
            single_mask_draw = ImageDraw.Draw(single_mask_layer)
            single_mask_draw.rectangle([x_min, y_min, x_max, y_max], 
                                      outline=outline_color, width=4, fill=None)
            
            # 計算質心
            centroid = get_centroid(mask_array)
            if centroid:
                cx, cy = centroid
                
                # 繪製質心點（稍大一點，更明顯）- 在單個 mask 圖層上
                single_mask_draw.ellipse([cx-6, cy-6, cx+6, cy+6], 
                                        fill=outline_color, outline=None)
                
                # 標註文字
                # 使用 mask_name 格式（例如 "buffer_0", "pallet_1"）
                # 這樣 VLM 可以更清楚地識別物體類型
                label = mask_name
                
                # 計算文字大小
                bbox = draw.textbbox((0, 0), label, font=dynamic_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 文字位置（在質心上方）
                text_x = cx - text_width // 2
                text_y = cy - text_height - 12
                
                # 確保文字不超出圖像邊界
                text_x = max(5, min(text_x, img.width - text_width - 5))
                text_y = max(5, min(text_y, img.height - text_height - 5))
                
                # 繪製文字背景（白色高不透明度，確保對比度）
                padding = 4
                draw.rectangle(
                    [text_x - padding, text_y - padding, 
                     text_x + text_width + padding, text_y + text_height + padding],
                    fill=(255, 255, 255, 230),  # 高不透明度白色背景
                    outline=outline_color,  # 邊框顏色跟 mask 一致
                    width=2
                )
                
                # 繪製文字（黑色，確保在白色背景上的高對比度）
                text_color = (0, 0, 0, 255)  # 黑色，完全不透明
                draw.text((text_x, text_y), label, fill=text_color, font=dynamic_font)
            
            # 合併單個 mask 圖層到總 overlay
            mask_overlay = Image.alpha_composite(mask_overlay, single_mask_layer)
    
    # 將所有 masks 的 overlay 合成到主圖像
    annotated_img = Image.alpha_composite(
        annotated_img.convert('RGBA'),
        mask_overlay
    ).convert('RGB')
    
    # 保存標註後的圖像
    annotated_img.save(output_path, quality=95)
    return annotated_img


def main():
    parser = argparse.ArgumentParser(description='在圖像上標註物件代號')
    parser.add_argument('--json_file', type=str, 
                      default='data/val.json',
                      help='JSON 數據文件路徑')
    parser.add_argument('--image_dir', type=str,
                      default='data/val/images',
                      help='圖像目錄路徑')
    parser.add_argument('--output_dir', type=str,
                      default='data/val/images_annotated',
                      help='輸出目錄路徑')
    parser.add_argument('--limit', type=int, default=None,
                      help='處理的圖像數量限制（用於測試）')
    parser.add_argument('--font_size', type=int, default=30,
                      help='字體大小')
    parser.add_argument('--start_idx', type=int, default=0,
                      help='開始索引（用於分批處理）')
    parser.add_argument('--use_region_id', action='store_true', default=True,
                      help='使用 Region ID 格式標註（如 "Region 0"），否則使用 mask_name 格式（如 "pallet_0"）')
    parser.add_argument('--skip_existing', action='store_true', default=False,
                      help='跳過已存在的標註圖像（不重新生成）')
    parser.add_argument('--force', action='store_true', default=False,
                      help='強制重新標註所有圖像（即使已存在）')
    
    # LLM 相關參數
    parser.add_argument('--use_llm', action='store_true', default=False,
                      help='使用 LLM 進行 mask 類型分類（否則使用 rule-based）')
    parser.add_argument('--llm_type', type=str, default='vllm',
                      choices=['gemini', 'vllm'],
                      help='LLM 類型: gemini 或 vllm')
    parser.add_argument('--api_base', type=str, default=None,
                      help='API base URL（用於 vLLM）')
    parser.add_argument('--api_key', type=str, default='dummy',
                      help='API key（用於 vLLM，可選）')
    parser.add_argument('--model_name', type=str, default=None,
                      help='模型名稱（用於 vLLM）')
    parser.add_argument('--project_id', type=str, default=None,
                      help='Project ID（用於 Gemini）')
    parser.add_argument('--location', type=str, default='global',
                      help='Location（用於 Gemini）')
    parser.add_argument('--temperature', type=float, default=0.2,
                      help='Temperature 參數')
    parser.add_argument('--max_tokens', type=int, default=2048,
                      help='最大 token 數')
    
    args = parser.parse_args()
    
    # 如果使用 LLM，創建 LLM client
    llm_client = None
    if args.use_llm:
        if args.llm_type == 'gemini':
            if not args.project_id:
                raise ValueError("--project_id is required when using Gemini")
            llm_client = create_llm_client(
                client_type='gemini',
                project_id=args.project_id,
                location=args.location
            )
            print("使用 Gemini (Vertex AI) 進行 mask 分類")
        elif args.llm_type == 'vllm':
            if not args.api_base or not args.model_name:
                raise ValueError("--api_base and --model_name are required when using vLLM")
            llm_client = create_llm_client(
                client_type='vllm',
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                is_vision_model=False  # mask 分類不需要視覺模型
            )
            print(f"使用 vLLM 進行 mask 分類: {args.model_name}")
        else:
            raise ValueError(f"不支援的 LLM 類型: {args.llm_type}")
    else:
        print("使用 rule-based 方法進行 mask 分類")
    
    # 讀取 JSON 文件
    print(f"讀取 JSON 文件: {args.json_file}")
    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"總共 {len(data)} 個項目")
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 處理每個項目
    processed = 0
    skipped = 0
    errors = []
    
    # 確定處理範圍
    end_idx = len(data) if args.limit is None else min(args.start_idx + args.limit, len(data))
    items_to_process = data[args.start_idx:end_idx]
    
    print(f"處理項目 {args.start_idx} 到 {end_idx-1} (共 {len(items_to_process)} 個)")
    
    for item in tqdm(items_to_process, desc="標註圖像"):
        try:
            item_id = item.get('id', 'unknown')
            image_filename = item.get('image', '')
            
            if not image_filename:
                skipped += 1
                continue
            
            # 構建圖像路徑
            image_path = os.path.join(args.image_dir, image_filename)
            
            if not os.path.exists(image_path):
                print(f"\n警告: 圖像不存在: {image_path}")
                skipped += 1
                continue
            
            # 解析 masks（如果使用 LLM，傳入 llm_client）
            conversation = item['conversations'][0]['value']
            rle_data = item['rle']
            
            masks_dict = parse_masks_from_conversation(conversation, rle_data, llm_client=llm_client)
            
            if not masks_dict:
                print(f"\n警告: 無法解析 masks for {item_id}")
                skipped += 1
                continue
            
            # 調試：確認 masks 數量
            if processed < 3:  # 只對前3個輸出調試信息
                print(f"\n[調試] {item_id}: 解析到 {len(masks_dict)} 個 masks")
                for mask_name in masks_dict.keys():
                    print(f"  - {mask_name}")
            
            # 構建輸出路徑
            base_name = os.path.splitext(image_filename)[0]
            output_filename = f"{base_name}_annotated.png"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # 檢查是否跳過已存在的文件
            if not args.force and args.skip_existing and os.path.exists(output_path):
                skipped += 1
                continue
            
            # 標註圖像
            annotate_image(image_path, masks_dict, output_path, 
                         font_size=args.font_size, 
                         use_region_id=args.use_region_id)
            
            processed += 1
            
        except Exception as e:
            error_msg = f"處理 {item.get('id', 'unknown')} 時出錯: {str(e)}"
            print(f"\n錯誤: {error_msg}")
            errors.append(error_msg)
            skipped += 1
    
    # 輸出統計信息
    print(f"\n處理完成!")
    print(f"成功處理: {processed} 個圖像")
    print(f"跳過/錯誤: {skipped} 個")
    print(f"輸出目錄: {args.output_dir}")
    
    if errors:
        print(f"\n錯誤列表 (前10個):")
        for err in errors[:10]:
            print(f"  - {err}")
    
    print(f"\n標註後的圖像已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

