#!/usr/bin/env python3
"""
快速查看特定 query 的 mask 編號對應關係
用法: python show_mask_labels.py <item_id> 或 python show_mask_labels.py --query <問題文本>
"""

import json
import os
import sys
import argparse
from mask import parse_masks_from_conversation

def show_masks_for_item(item_id=None, query_text=None, results_file='../output/train_eval_results.json'):
    """顯示指定 item 的 mask 信息"""
    
    # 載入 train.json
    train_json_path = '../data/train/train.json'
    if not os.path.exists(train_json_path):
        print(f"錯誤: 找不到 {train_json_path}")
        return
    
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    # 找到對應的 item
    target_item = None
    if item_id:
        for item in train_data:
            if item['id'] == item_id:
                target_item = item
                break
    elif query_text:
        for item in train_data:
            conversation = item['conversations'][0]['value']
            if query_text.lower() in conversation.lower():
                target_item = item
                break
    
    if not target_item:
        print("找不到對應的 item")
        return
    
    item_id = target_item['id']
    conversation = target_item['conversations'][0]['value']
    rle_data = target_item['rle']
    image_name = target_item['image']
    
    # 解析 masks
    masks_dict = parse_masks_from_conversation(conversation, rle_data)
    
    print("=" * 80)
    print(f"Item ID: {item_id}")
    print(f"Image: {image_name}")
    print("=" * 80)
    print(f"\n問題:")
    print(f"  {conversation}")
    print(f"\n找到 {len(masks_dict)} 個 masks:")
    print("-" * 80)
    
    # 按類別分組顯示
    from collections import defaultdict
    masks_by_class = defaultdict(list)
    for mask_name, mask_obj in masks_dict.items():
        masks_by_class[mask_obj.object_class].append((mask_name, mask_obj))
    
    for object_class in sorted(masks_by_class.keys()):
        print(f"\n{object_class.upper()}:")
        for mask_name, mask_obj in sorted(masks_by_class[object_class], 
                                          key=lambda x: x[1].object_id):
            print(f"  {mask_name:15s} -> region_id={mask_obj.region_id:2d}, "
                  f"object_id={mask_obj.object_id}, "
                  f"RLE size={mask_obj.rle.get('size', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("可視化命令:")
    print(f"  python visualize_masks_with_labels.py --item_ids {item_id}")
    print("=" * 80)
    
    # 如果有結果文件，顯示預測結果
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        for result in results:
            if result.get('id') == item_id:
                print(f"\n預測結果:")
                print(f"  預測: {result.get('predicted', 'N/A')}")
                print(f"  正確答案: {result.get('ground_truth', 'N/A')}")
                print(f"  正確: {result.get('correct', False)}")
                print(f"  類別: {result.get('category', 'unknown')}")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='查看 mask 編號對應關係')
    parser.add_argument('item_id', nargs='?', help='Item ID')
    parser.add_argument('--query', type=str, help='問題文本（部分匹配）')
    parser.add_argument('--results_file', type=str, 
                       default='../output/train_eval_results.json',
                       help='結果文件路徑')
    
    args = parser.parse_args()
    
    if not args.item_id and not args.query:
        print("請提供 item_id 或 --query 參數")
        print("\n示例:")
        print("  python show_mask_labels.py 92ba42f4dc21d0b51424aa1b07508536")
        print("  python show_mask_labels.py --query 'buffer region'")
        sys.exit(1)
    
    show_masks_for_item(
        item_id=args.item_id,
        query_text=args.query,
        results_file=args.results_file
    )

