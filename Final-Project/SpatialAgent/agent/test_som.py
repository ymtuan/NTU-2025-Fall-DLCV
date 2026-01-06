#!/usr/bin/env python3
"""
Set-of-Marks (SoM) 測試腳本
演示如何使用 SoM 可視化器
"""

import os
import sys
import json
from som_visualizer import SoMVisualizer
from mask import parse_masks_from_conversation

def test_som_with_sample():
    """使用訓練集樣本測試 SoM 可視化"""
    
    # 載入訓練集數據
    train_data_path = '../data/train/train.json'
    if not os.path.exists(train_data_path):
        print(f"錯誤：找不到訓練集文件 {train_data_path}")
        return False
    
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
    
    # 找第一個有效的樣本
    sample_item = None
    for item in train_data:
        image_path = f"../data/train/images/{item['image']}"
        if os.path.exists(image_path):
            sample_item = item
            break
    
    if sample_item is None:
        print("錯誤：找不到有效的訓練樣本")
        return False
    
    print(f"使用樣本: {sample_item['id']}")
    print(f"圖像: {sample_item['image']}")
    print(f"問題: {sample_item['conversations'][0]['value'][:200]}...")
    
    # 解析 masks
    conversation = sample_item['conversations'][0]['value']
    rle_data = sample_item['rle']
    masks = parse_masks_from_conversation(conversation, rle_data)
    
    print(f"找到 {len(masks)} 個 masks:")
    for mask_name, mask_obj in masks.items():
        print(f"  - {mask_name}: {mask_obj}")
    
    # 創建 SoM 可視化器
    som_viz = SoMVisualizer(font_size=30, mask_alpha=120)
    
    # 創建輸出目錄
    output_dir = '../output/som_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # 測試不同的編號方案
    schemes = ['region_id', 'object_id', 'sequential']
    
    for scheme in schemes:
        print(f"\n測試編號方案: {scheme}")
        
        # 生成可視化
        image_path = f"../data/train/images/{sample_item['image']}"
        som_output = os.path.join(output_dir, f"som_{scheme}_{sample_item['id']}.png")
        legend_output = os.path.join(output_dir, f"legend_{scheme}_{sample_item['id']}.png")
        
        try:
            som_viz.create_som_visualization(
                image_path=image_path,
                masks=masks,
                output_path=som_output,
                numbering_scheme=scheme
            )
            
            som_viz.create_legend(
                masks=masks,
                output_path=legend_output,
                numbering_scheme=scheme
            )
            
            print(f"  ✓ SoM 可視化: {som_output}")
            print(f"  ✓ 圖例: {legend_output}")
            
        except Exception as e:
            print(f"  ✗ 錯誤: {e}")
    
    print(f"\n所有測試文件已保存到: {output_dir}")
    return True

def demonstrate_som_features():
    """展示 SoM 的主要功能"""
    
    print("=" * 60)
    print("Set-of-Marks (SoM) 可視化器功能展示")
    print("=" * 60)
    
    print("\n1. 主要功能:")
    print("   - 半透明彩色 mask 覆蓋")
    print("   - 清晰的數字標籤")
    print("   - 多種編號方案")
    print("   - 自動圖例生成")
    
    print("\n2. 編號方案:")
    print("   - region_id: 使用 mask 在 RLE 數據中的索引（0, 1, 2, ...）")
    print("   - object_id: 使用物件類別縮寫+編號（P0, B1, T0, ...）")
    print("   - sequential: 順序編號（0, 1, 2, ...）")
    
    print("\n3. 可調參數:")
    print("   - font_size: 標籤字體大小（預設 30）")
    print("   - mask_alpha: mask 透明度 0-255（預設 120）")
    print("   - font_path: 自定義字體路徑")
    
    print("\n4. 輸出文件:")
    print("   - som_[scheme]_[id].png: SoM 可視化圖像")
    print("   - legend_[scheme]_[id].png: 對應的圖例")
    
    print("\n5. 使用方式:")
    print("   python train_eval.py --enable_som --som_scheme region_id")

def main():
    """主測試函數"""
    
    print("SoM 可視化器測試")
    print("=" * 40)
    
    # 檢查必要文件
    required_files = [
        '../data/train/train.json',
        'mask.py',
        'som_visualizer.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"錯誤：缺少必要文件:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    
    # 展示功能
    demonstrate_som_features()
    
    # 運行測試
    print("\n" + "=" * 60)
    print("運行測試...")
    print("=" * 60)
    
    success = test_som_with_sample()
    
    if success:
        print("\n✓ SoM 測試完成！")
        print("\n後續步驟:")
        print("1. 檢查生成的可視化文件")
        print("2. 運行完整的 train_eval.py 並啟用 SoM")
        print("3. 比較不同編號方案的效果")
        return True
    else:
        print("\n✗ SoM 測試失敗")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)