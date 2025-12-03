#!/usr/bin/env python3
"""
Set-of-Marks (SoM) Visualizer
在物體上使用半透明 mask 覆蓋，並壓上清晰的數字標籤
"""

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Dict, List, Optional, Tuple
from mask import Mask


class SoMVisualizer:
    """Set-of-Marks 可視化器"""
    
    def __init__(self, 
                 font_size: int = 30,
                 mask_alpha: int = 120,  # 半透明度 (0-255)
                 font_path: Optional[str] = None):
        """
        初始化 SoM 可視化器
        
        Args:
            font_size: 標籤字體大小
            mask_alpha: Mask 半透明度 (0完全透明, 255完全不透明)
            font_path: 自定義字體路徑
        """
        self.font_size = font_size
        self.mask_alpha = mask_alpha
        
        # 設置字體
        if font_path and os.path.exists(font_path):
            self.font = ImageFont.truetype(font_path, font_size)
        else:
            # 嘗試常見的系統字體
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Arial.ttf",
                "/Windows/Fonts/arial.ttf",
            ]
            self.font = None
            for fp in font_paths:
                if os.path.exists(fp):
                    self.font = ImageFont.truetype(fp, font_size)
                    break
            
            if self.font is None:
                self.font = ImageFont.load_default()
        
        # 預定義顏色列表（避免相鄰物件顏色太相似）
        self.colors = [
            (255, 0, 0),    # 紅色
            (0, 255, 0),    # 綠色
            (0, 0, 255),    # 藍色
            (255, 255, 0),  # 黃色
            (255, 0, 255),  # 洋紅
            (0, 255, 255),  # 青色
            (255, 128, 0),  # 橙色
            (128, 0, 255),  # 紫色
            (255, 0, 128),  # 粉紅
            (128, 255, 0),  # 萊姆綠
            (0, 128, 255),  # 天藍色
            (255, 128, 128), # 淺紅
            (128, 255, 128), # 淺綠
            (128, 128, 255), # 淺藍
        ]
    
    def create_som_visualization(self, 
                               image_path: str,
                               masks: Dict[str, Mask],
                               output_path: str,
                               numbering_scheme: str = "region_id") -> str:
        """
        創建 Set-of-Marks 可視化圖像
        
        Args:
            image_path: 輸入圖像路徑
            masks: Mask 字典，key 為 mask 名稱
            output_path: 輸出圖像路徑
            numbering_scheme: 編號方案 ("region_id", "object_id", "sequential")
        
        Returns:
            輸出圖像路徑
        """
        # 載入原始圖像
        image = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        
        # 準備標籤信息
        label_infos = []
        
        # 根據編號方案對 masks 排序
        sorted_masks = self._sort_masks_by_scheme(masks, numbering_scheme)
        
        for idx, (mask_name, mask_obj) in enumerate(sorted_masks):
            # 解碼 mask
            mask_array = mask_obj.decode_mask()
            
            if mask_array.sum() == 0:  # 跳過空 mask
                continue
                
            # 創建半透明彩色 mask
            color = self.colors[idx % len(self.colors)]
            colored_mask = self._create_colored_mask(
                mask_array, color, image.size, self.mask_alpha
            )
            
            # 添加到覆蓋層
            overlay = Image.alpha_composite(overlay, colored_mask)
            
            # 計算標籤位置和內容
            label_text, label_pos = self._calculate_label_position(
                mask_array, mask_obj, idx, numbering_scheme
            )
            
            label_infos.append((label_text, label_pos, color))
        
        # 合成最終圖像
        final_image = Image.alpha_composite(image, overlay)
        
        # 添加標籤文字
        self._add_text_labels(final_image, label_infos)
        
        # 轉為 RGB 並保存
        final_rgb = final_image.convert("RGB")
        final_rgb.save(output_path)
        
        return output_path
    
    def _sort_masks_by_scheme(self, masks: Dict[str, Mask], scheme: str) -> List[Tuple[str, Mask]]:
        """根據指定方案對 masks 排序"""
        mask_list = list(masks.items())
        
        if scheme == "region_id":
            return sorted(mask_list, key=lambda x: x[1].region_id)
        elif scheme == "object_id":
            return sorted(mask_list, key=lambda x: (x[1].object_class, x[1].object_id))
        elif scheme == "sequential":
            return mask_list  # 保持原順序
        else:
            raise ValueError(f"未支援的編號方案: {scheme}")
    
    def _create_colored_mask(self, 
                           mask_array: np.ndarray, 
                           color: Tuple[int, int, int],
                           image_size: Tuple[int, int],
                           alpha: int) -> Image.Image:
        """創建彩色半透明 mask"""
        # 創建 mask 圖像
        mask_image = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
        
        # 創建彩色覆蓋
        colored_overlay = Image.new("RGBA", image_size, (*color, alpha))
        
        # 應用 mask
        result = Image.new("RGBA", image_size, (0, 0, 0, 0))
        result.paste(colored_overlay, (0, 0), mask_image)
        
        return result
    
    def _calculate_label_position(self, 
                                mask_array: np.ndarray,
                                mask_obj: Mask,
                                index: int,
                                scheme: str) -> Tuple[str, Tuple[int, int]]:
        """計算標籤文字和位置"""
        # 確定標籤文字 - 統一使用 object_class_object_id 格式
        label_text = f"{mask_obj.object_class}_{mask_obj.object_id}"
        
        # 計算 mask 的質心位置
        mask_indices = np.argwhere(mask_array)
        if mask_indices.size > 0:
            center_y = int(np.mean(mask_indices[:, 0]))
            center_x = int(np.mean(mask_indices[:, 1]))
        else:
            center_x, center_y = 0, 0
        
        return label_text, (center_x, center_y)
    
    def _add_text_labels(self, 
                        image: Image.Image, 
                        label_infos: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]]):
        """添加文字標籤"""
        draw = ImageDraw.Draw(image)
        
        for text, (x, y), color in label_infos:
            # 計算文字尺寸
            bbox = draw.textbbox((0, 0), text, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 調整位置使文字居中
            text_x = x - text_width // 2
            text_y = y - text_height // 2
            
            # 繪製文字背景（提高可讀性）
            padding = 3
            bg_bbox = [
                text_x - padding, 
                text_y - padding,
                text_x + text_width + padding,
                text_y + text_height + padding
            ]
            draw.rectangle(bg_bbox, fill=(0, 0, 0, 180))
            
            # 繪製文字
            draw.text((text_x, text_y), text, fill=(255, 255, 255), font=self.font)
    
    def create_legend(self, 
                     masks: Dict[str, Mask], 
                     output_path: str,
                     numbering_scheme: str = "region_id") -> str:
        """創建圖例，顯示數字標籤對應的物件信息"""
        sorted_masks = self._sort_masks_by_scheme(masks, numbering_scheme)
        
        # 計算圖例尺寸
        legend_width = 400
        line_height = 40
        legend_height = max(200, len(sorted_masks) * line_height + 60)
        
        # 創建圖例圖像
        legend_img = Image.new("RGB", (legend_width, legend_height), (255, 255, 255))
        draw = ImageDraw.Draw(legend_img)
        
        # 標題
        title = f"Set-of-Marks Legend ({numbering_scheme})"
        draw.text((10, 10), title, fill=(0, 0, 0), font=self.font)
        
        # 圖例項目
        for idx, (mask_name, mask_obj) in enumerate(sorted_masks):
            y_pos = 50 + idx * line_height
            color = self.colors[idx % len(self.colors)]
            
            # 顏色方塊
            draw.rectangle([10, y_pos, 30, y_pos + 20], fill=color)
            
            # 標籤文字
            if numbering_scheme == "region_id":
                label = str(mask_obj.region_id)
            elif numbering_scheme == "object_id":
                label = f"{mask_obj.object_class[0].upper()}{mask_obj.object_id}"
            else:
                label = str(idx)
            
            # 描述文字
            desc = f"{label}: {mask_obj.object_class}_{mask_obj.object_id}"
            draw.text((40, y_pos), desc, fill=(0, 0, 0), font=self.font)
        
        legend_img.save(output_path)
        return output_path


def enhance_train_eval_with_som():
    """為 train_eval.py 添加 SoM 可視化功能的示例代碼"""
    
    # 這是一個示例，展示如何在 train_eval.py 中集成 SoM
    example_code = '''
# 在 train_eval.py 的 main() 函數中添加以下代碼：

# 導入 SoM 可視化器
from som_visualizer import SoMVisualizer

# 在處理循環中添加可視化
som_viz = SoMVisualizer(font_size=25, mask_alpha=100)

for item in tqdm(valid_data, desc="Processing"):
    try:
        agent = Agent(llm_client, tools, item, think_mode=args.think_mode, verbose=args.verbose)
        agent.set_masks()
        
        # 創建 SoM 可視化（可選）
        if args.save_visualizations:  # 新增命令行參數
            image_path = f"../data/train/images/{item['image']}"
            som_output = f"../output/som_visualizations/som_{item['id']}.png"
            legend_output = f"../output/som_visualizations/legend_{item['id']}.png"
            
            os.makedirs("../output/som_visualizations", exist_ok=True)
            
            # 生成 SoM 可視化
            som_viz.create_som_visualization(
                image_path=image_path,
                masks=agent.masks,
                output_path=som_output,
                numbering_scheme="region_id"  # 或 "object_id"
            )
            
            # 生成圖例
            som_viz.create_legend(
                masks=agent.masks,
                output_path=legend_output,
                numbering_scheme="region_id"
            )
        
        answer = agent.set_question()
        answer = agent.format_answer()
        
        # 其餘處理邏輯...
        
    except Exception as e:
        # 錯誤處理...
'''
    
    return example_code


if __name__ == "__main__":
    # 測試代碼
    print("SoM Visualizer 模組已創建")
    print("\n使用方式：")
    print("1. 導入模組：from som_visualizer import SoMVisualizer")
    print("2. 創建實例：som_viz = SoMVisualizer()")
    print("3. 生成可視化：som_viz.create_som_visualization(image_path, masks, output_path)")
    print("4. 生成圖例：som_viz.create_legend(masks, legend_path)")
    
    print("\n" + "="*50)
    print("集成到 train_eval.py 的示例代碼：")
    print(enhance_train_eval_with_som())