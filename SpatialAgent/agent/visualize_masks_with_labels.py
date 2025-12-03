#!/usr/bin/env python3
"""
可視化 masks 並標註編號（如 pallet_0, buffer_1 等）
用於查看每個 mask 編號對應到圖像上的哪個區域
"""

import json
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pycocotools.mask as mask_utils
from collections import defaultdict
import random

def get_color_for_mask(mask_name):
    """為每個 mask 生成固定顏色"""
    # 使用 hash 來生成固定顏色
    hash_val = hash(mask_name)
    random.seed(hash_val)
    r = random.randint(50, 255)
    g = random.randint(50, 255)
    b = random.randint(50, 255)
    random.seed()  # 重置隨機種子
    return (r, g, b, 180)  # 180 是透明度

def visualize_masks_with_labels(image_path, masks_dict, output_path, show_legend=True):
    """
    可視化 masks 並標註名稱
    
    Args:
        image_path: 圖像路徑
        masks_dict: Dict[str, Mask] - mask 名稱到 Mask 對象的映射
        output_path: 輸出圖像路徑
        show_legend: 是否顯示圖例
    """
    # 載入圖像
    if not os.path.exists(image_path):
        print(f"警告: 圖像文件不存在: {image_path}")
        return False
    
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    
    # 嘗試載入字體
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
    ]
    font = None
    font_size = 20
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except:
                pass
    
    if font is None:
        font = ImageFont.load_default()
    
    # 準備文本信息
    text_infos = []
    legend_items = []
    
    # 按 object_class 分組，方便排序
    masks_by_class = defaultdict(list)
    for mask_name, mask_obj in masks_dict.items():
        masks_by_class[mask_obj.object_class].append((mask_name, mask_obj))
    
    # 處理每個 mask
    for object_class in sorted(masks_by_class.keys()):
        for mask_name, mask_obj in sorted(masks_by_class[object_class], 
                                          key=lambda x: x[1].object_id):
            try:
                # 解碼 mask
                mask_array = mask_obj.decode_mask()
                
                # 生成顏色
                color = get_color_for_mask(mask_name)
                
                # 創建彩色 mask
                mask_image = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
                colored_mask = Image.new("RGBA", image.size, color)
                overlay.paste(colored_mask, (0, 0), mask_image)
                
                # 計算 mask 的中心位置用於標註
                mask_indices = np.argwhere(mask_array)
                if mask_indices.size > 0:
                    min_y, min_x = mask_indices.min(axis=0)
                    max_y, max_x = mask_indices.max(axis=0)
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2
                    
                    # 標註文本
                    label = mask_name  # 例如 "pallet_0"
                    
                    # 計算文本大小
                    draw_temp = ImageDraw.Draw(overlay)
                    bbox = draw_temp.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # 文本位置（在 mask 中心）
                    text_position = (center_x - text_width // 2, center_y - text_height // 2)
                    text_infos.append((label, text_position, color))
                    
                    # 添加到圖例
                    legend_items.append((label, color))
                    
            except Exception as e:
                print(f"處理 mask {mask_name} 時出錯: {e}")
                continue
    
    # 繪製文本標註
    draw = ImageDraw.Draw(overlay)
    for label, text_position, color in text_infos:
        # 繪製文本背景（白色半透明）
        bbox = draw.textbbox(text_position, label, font=font)
        padding = 2
        bg_box = (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding
        )
        draw.rectangle(bg_box, fill=(255, 255, 255, 200))
        
        # 繪製文本
        draw.text(text_position, label, fill=(0, 0, 0, 255), font=font)
    
    # 混合圖像
    blended_image = Image.alpha_composite(image, overlay)
    
    # 如果需要顯示圖例
    if show_legend and legend_items:
        # 創建圖例
        legend_height = len(legend_items) * 30 + 40
        legend_width = 200
        legend_image = Image.new("RGBA", (legend_width, legend_height), (255, 255, 255, 240))
        legend_draw = ImageDraw.Draw(legend_image)
        
        # 標題
        legend_draw.text((10, 10), "Masks:", fill=(0, 0, 0, 255), font=font)
        
        # 每個 mask 的圖例項
        y_offset = 40
        for label, color in legend_items:
            # 顏色方塊
            legend_draw.rectangle([10, y_offset, 30, y_offset + 15], fill=color[:3])
            # 標籤文本
            legend_draw.text((35, y_offset - 2), label, fill=(0, 0, 0, 255), font=font)
            y_offset += 25
        
        # 將圖例添加到右側
        final_width = blended_image.width + legend_width
        final_height = max(blended_image.height, legend_height)
        final_image = Image.new("RGBA", (final_width, final_height), (255, 255, 255, 255))
        final_image.paste(blended_image, (0, 0))
        final_image.paste(legend_image, (blended_image.width, 0))
        blended_image = final_image
    
    # 保存圖像
    blended_image.convert("RGB").save(output_path)
    print(f"已保存可視化圖像: {output_path}")
    return True

def visualize_from_train_eval_results(results_file, output_dir, item_ids=None, max_items=10):
    """
    從 train_eval.py 的結果文件中讀取數據並可視化
    
    Args:
        results_file: train_eval_results.json 文件路徑
        output_dir: 輸出目錄
        item_ids: 可選，指定要可視化的 item ID 列表
        max_items: 最多可視化多少個項目
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 過濾要可視化的項目
    if item_ids:
        results = [r for r in results if r.get('id') in item_ids]
    else:
        results = results[:max_items]
    
    print(f"準備可視化 {len(results)} 個項目...")
    
    # 需要從原始 train.json 載入 RLE 數據
    train_json_path = '../data/train/train.json'
    if not os.path.exists(train_json_path):
        print(f"錯誤: 找不到 train.json: {train_json_path}")
        return
    
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    # 創建 ID 到數據的映射
    train_dict = {item['id']: item for item in train_data}
    
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from mask import parse_masks_from_conversation
    
    for result in results:
        item_id = result.get('id')
        if item_id not in train_dict:
            print(f"警告: 找不到 item {item_id} 的數據")
            continue
        
        train_item = train_dict[item_id]
        image_name = train_item['image']
        image_path = f"../data/train/images/{image_name}"
        
        # 解析 masks
        conversation = train_item['conversations'][0]['value']
        rle_data = train_item['rle']
        masks_dict = parse_masks_from_conversation(conversation, rle_data)
        
        if not masks_dict:
            print(f"警告: item {item_id} 沒有找到 masks")
            continue
        
        # 可視化
        output_path = os.path.join(output_dir, f"{item_id}_{image_name}")
        visualize_masks_with_labels(image_path, masks_dict, output_path, show_legend=True)
        
        # 打印 mask 信息
        print(f"\nItem {item_id}:")
        print(f"  問題: {conversation[:100]}...")
        print(f"  Masks: {', '.join(sorted(masks_dict.keys()))}")
        print(f"  預測: {result.get('predicted', 'N/A')}")
        print(f"  正確答案: {result.get('ground_truth', 'N/A')}")
        print(f"  正確: {result.get('correct', False)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可視化 masks 並標註編號')
    parser.add_argument('--results_file', type=str, 
                       default='../output/train_eval_results.json',
                       help='train_eval.py 的結果文件')
    parser.add_argument('--output_dir', type=str,
                       default='../output/mask_visualizations',
                       help='輸出目錄')
    parser.add_argument('--item_ids', type=str, nargs='+',
                       help='指定要可視化的 item ID 列表')
    parser.add_argument('--max_items', type=int, default=10,
                       help='最多可視化多少個項目')
    parser.add_argument('--image_path', type=str,
                       help='直接可視化單個圖像（需要配合 --masks_json）')
    parser.add_argument('--masks_json', type=str,
                       help='masks 的 JSON 文件（用於單個圖像可視化）')
    
    args = parser.parse_args()
    
    if args.image_path and args.masks_json:
        # 單個圖像可視化模式
        with open(args.masks_json, 'r') as f:
            masks_data = json.load(f)
        
        # 假設 masks_data 是一個字典，key 是 mask 名稱，value 包含 Mask 對象的信息
        # 這裡需要根據實際格式調整
        print("單個圖像可視化模式需要根據實際數據格式調整")
    else:
        # 從結果文件可視化
        visualize_from_train_eval_results(
            args.results_file,
            args.output_dir,
            item_ids=args.item_ids,
            max_items=args.max_items
        )

