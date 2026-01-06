#!/usr/bin/env python3
"""
分析 train set 評估結果的錯誤
"""

import json
import os
import sys
import argparse
from collections import Counter, defaultdict
import re

def load_results(results_file):
    """載入評估結果"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def analyze_errors(results):
    """分析錯誤"""
    errors = [r for r in results if not r.get('correct', False)]
    correct = [r for r in results if r.get('correct', False)]
    
    print("=" * 80)
    print("錯誤分析報告")
    print("=" * 80)
    
    print(f"\n總體統計:")
    print(f"  總數: {len(results)}")
    print(f"  正確: {len(correct)} ({len(correct)/len(results)*100:.1f}%)")
    print(f"  錯誤: {len(errors)} ({len(errors)/len(results)*100:.1f}%)")
    
    # 按類別分析
    print(f"\n按類別統計:")
    category_stats = defaultdict(lambda: {'correct': 0, 'error': 0, 'total': 0})
    
    for r in results:
        cat = r.get('category', 'unknown')
        category_stats[cat]['total'] += 1
        if r.get('correct', False):
            category_stats[cat]['correct'] += 1
        else:
            category_stats[cat]['error'] += 1
    
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {cat}:")
        print(f"    正確: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        print(f"    錯誤: {stats['error']}/{stats['total']} ({100-acc:.1f}%)")
    
    # 錯誤類型分析
    print(f"\n錯誤類型分析:")
    error_types = Counter()
    for err in errors:
        if 'error' in err:
            error_msg = err['error']
            if 'decode_mask' in error_msg:
                error_types['mask_decode_error'] += 1
            elif 'No valid action' in error_msg:
                error_types['no_valid_action'] += 1
            elif 'Mask' in error_msg and 'not found' in error_msg:
                error_types['mask_not_found'] += 1
            elif 'Invalid function' in error_msg:
                error_types['invalid_function'] += 1
            else:
                error_types['other_error'] += 1
        else:
            # 預測錯誤（不是異常）
            pred = str(err.get('predicted', '')).lower()
            gt = str(err.get('ground_truth', '')).lower()
            
            # 分析錯誤模式
            try:
                pred_num = float(pred)
                gt_num = float(gt)
                diff = abs(pred_num - gt_num)
                if diff < 0.5:
                    error_types['small_numerical_error'] += 1
                elif diff < 2.0:
                    error_types['medium_numerical_error'] += 1
                else:
                    error_types['large_numerical_error'] += 1
            except:
                if pred == '-1':
                    error_types['prediction_failed'] += 1
                elif pred != gt:
                    error_types['wrong_answer'] += 1
    
    for err_type, count in error_types.most_common():
        print(f"  {err_type}: {count} 個")
    
    # 常見錯誤模式
    print(f"\n常見錯誤模式（預測 vs 正確答案）:")
    error_patterns = Counter()
    for err in errors[:100]:  # 只分析前100個避免太多
        if 'error' not in err:
            pred = str(err.get('predicted', '')).lower()
            gt = str(err.get('ground_truth', '')).lower()
            pattern = f"pred={pred}, gt={gt}"
            error_patterns[pattern] += 1
    
    for pattern, count in error_patterns.most_common(10):
        print(f"  {pattern}: {count} 次")
    
    # 找出最需要關注的錯誤案例
    print(f"\n需要關注的錯誤案例:")
    
    # 1. 有異常的錯誤
    exception_errors = [e for e in errors if 'error' in e]
    if exception_errors:
        print(f"\n  異常錯誤 ({len(exception_errors)} 個):")
        for i, err in enumerate(exception_errors[:5], 1):
            print(f"    {i}. ID: {err['id']}")
            print(f"       錯誤: {err['error'][:100]}")
            print(f"       類別: {err.get('category', 'N/A')}")
    
    # 2. 距離問題的大誤差
    distance_errors = [e for e in errors if e.get('category') == 'distance' and 'error' not in e]
    if distance_errors:
        large_errors = []
        for err in distance_errors:
            try:
                pred = float(err.get('predicted', 0))
                gt = float(err.get('ground_truth', 0))
                diff = abs(pred - gt)
                large_errors.append((diff, err))
            except:
                pass
        
        large_errors.sort(reverse=True, key=lambda x: x[0])
        print(f"\n  距離問題大誤差 (前5個):")
        for i, (diff, err) in enumerate(large_errors[:5], 1):
            print(f"    {i}. ID: {err['id']}")
            print(f"       預測: {err.get('predicted')}, 正確: {err.get('ground_truth')}, 誤差: {diff:.2f}m")
    
    # 3. Count 問題的錯誤
    count_errors = [e for e in errors if e.get('category') == 'count' and 'error' not in e]
    if count_errors:
        print(f"\n  Count 問題錯誤 (前5個):")
        for i, err in enumerate(count_errors[:5], 1):
            print(f"    {i}. ID: {err['id']}")
            print(f"       預測: {err.get('predicted')}, 正確: {err.get('ground_truth')}")
    
    return errors

def visualize_error_cases(errors, train_data_file, output_dir='../output/error_visualization'):
    """可視化錯誤案例"""
    import sys
    sys.path.insert(0, '../utils')
    from visualize import visualize_masks_and_depth
    
    # 載入 train data
    with open(train_data_file, 'r') as f:
        train_data = {item['id']: item for item in json.load(f)}
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n可視化錯誤案例...")
    visualized = 0
    
    for err in errors[:20]:  # 可視化前20個錯誤案例
        item_id = err['id']
        if item_id not in train_data:
            continue
        
        item = train_data[item_id]
        image_name = item.get('image', '')
        
        if not image_name:
            continue
        
        image_path = f"../data/train/images/{image_name}"
        depth_path = f"../data/train/depths/{image_name.replace('.png', '_depth.png')}"
        
        if not os.path.exists(image_path):
            continue
        
        # 準備輸出路徑
        output_path = os.path.join(output_dir, f"{item_id}_{image_name}")
        
        try:
            # 檢查 depth 文件是否存在
            if os.path.exists(depth_path):
                # 可視化
                visualize_masks_and_depth(
                    item.get('rle', []),
                    image_path,
                    depth_path,
                    output_path
                )
            else:
                # 只可視化 RGB + masks（沒有 depth）
                from PIL import Image, ImageDraw
                import pycocotools.mask as mask_utils
                import numpy as np
                import random
                
                image = Image.open(image_path).convert("RGBA")
                overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
                
                for i, rle in enumerate(item.get('rle', [])):
                    mask = mask_utils.decode(rle)
                    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 128)
                    colored_mask = Image.new("RGBA", image.size, color)
                    overlay.paste(colored_mask, (0, 0), mask_image)
                
                blended_image = Image.alpha_composite(image, overlay)
                blended_image.convert('RGB').save(output_path)
            
            visualized += 1
            
            # 保存錯誤信息
            info_file = output_path.replace('.png', '_info.txt')
            with open(info_file, 'w') as f:
                f.write(f"ID: {item_id}\n")
                f.write(f"Category: {err.get('category', 'N/A')}\n")
                f.write(f"Predicted: {err.get('predicted', 'N/A')}\n")
                f.write(f"Ground Truth: {err.get('ground_truth', 'N/A')}\n")
                if 'error' in err:
                    f.write(f"Error: {err['error']}\n")
                if 'conversation' in err:
                    f.write(f"\nConversation:\n")
                    for msg in err['conversation'][:5]:
                        f.write(f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:200]}\n")
        
        except Exception as e:
            print(f"  可視化失敗 {item_id}: {e}")
    
    print(f"✓ 已可視化 {visualized} 個錯誤案例到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="分析 train set 評估錯誤")
    parser.add_argument('--results_file', type=str, default='../output/train_eval_results.json',
                       help='評估結果文件')
    parser.add_argument('--train_data', type=str, default='../data/train/train.json',
                       help='Train data 文件')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可視化錯誤案例')
    parser.add_argument('--output_dir', type=str, default='../output/error_analysis',
                       help='輸出目錄')
    args = parser.parse_args()
    
    # 載入結果
    print("載入評估結果...")
    results = load_results(args.results_file)
    
    # 分析錯誤
    errors = analyze_errors(results)
    
    # 保存錯誤案例
    os.makedirs(args.output_dir, exist_ok=True)
    error_file = os.path.join(args.output_dir, 'detailed_errors.json')
    with open(error_file, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"\n✓ 詳細錯誤已保存到: {error_file}")
    
    # 可視化
    if args.visualize:
        visualize_error_cases(errors, args.train_data, 
                            os.path.join(args.output_dir, 'visualization'))

if __name__ == "__main__":
    main()

