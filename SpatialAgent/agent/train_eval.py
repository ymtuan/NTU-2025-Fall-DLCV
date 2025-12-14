#!/usr/bin/env python3
"""
在 train set 上評估並分析錯誤
"""

import re
import json
import os
import ast
import time
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict, Counter
from tools import tools_api
from mask import parse_masks_from_conversation
from llm_client import create_llm_client
from som_visualizer import SoMVisualizer

class Agent:
    def __init__(self, llm_client, tools_api, input_item, think_mode=False, verbose=False, som_visualizer=None, output_dir='../output', force_llm_category=False):
        self.llm_client = llm_client
        self.tools_api = tools_api
        self.verbose = verbose
        self.step_log = []  # 記錄每個步驟
        # self.som_visualizer = som_visualizer  # 已註解：SoM 功能暫時不使用
        self.output_dir = output_dir
        
        self.messages = []
        self.prompt_preamble = open('prompt/agent_example.txt', 'r').read()
        self.answer_preamble = open('prompt/answer.txt', 'r').read()
        self.input = input_item
        self.masks = None
        self.question = None
        # self.som_image_path = None  # 已註解：SoM 功能暫時不使用
        self.original_image_path = None  # 原始圖像路徑
        self.question_category = None  # 問題類別
        self._category_cache = {}  # 快取 LLM 分類結果
        self.force_llm_category = force_llm_category  # 強制使用 LLM 分類
        self.predicted_category = None  # LLM 預測的類別（用於評估）
        self.ground_truth_category = None  # 數據集提供的真實類別（用於準確率計算）
    
    def _detect_question_category_with_llm(self, question):
        """使用 LLM 檢測問題類別（僅 4 種類別）"""
        # 檢查快取
        if question in self._category_cache:
            return self._category_cache[question]
        
        # 跳過關鍵字檢測，直接使用 LLM（更準確）
        # keyword_category = self._detect_question_category(question)
        # if keyword_category != 'other':
        #     self._category_cache[question] = keyword_category
        #     return keyword_category
        
        # 使用 LLM 分類（僅 4 種類別）- 更詳細的 prompt
        classification_prompt = f"""Classify this spatial reasoning question into exactly ONE category.

Question: "{question}"

Categories (choose ONE):
1. left_right - Questions ONLY about determining if something is to the left OR right of another object. Examples:
   - "Is the pallet to the left of the buffer?"
   - "Which direction is the pallet from the buffer?"
   
2. count - Questions asking for a NUMBER or COUNT of objects. Examples:
   - "How many pallets are there?"
   - "What is the total number of pallets in the rightmost buffer?"
   - Note: Even if the question mentions "left/right/furthest", if it asks for a COUNT, it's still COUNT category
   
3. distance - Questions about DISTANCE or HOW FAR objects are. Examples:
   - "What is the distance between the pallet and buffer?"
   - "How far is the pallet from the buffer?"
   
4. mcq - Questions asking to SELECT/IDENTIFY WHICH specific object from a list. Examples:
   - "Which pallet is closest to the buffer?"
   - "Which of the pallets is positioned the furthest to the left?"
   - Note: These ask "WHICH" object, not "left or right"

Key distinction:
- "Is X left of Y?" → left_right (answer: yes/no or left/right)
- "Which of X, Y, Z is furthest left?" → mcq (answer: X or Y or Z)
- "How many in the leftmost region?" → count (answer: a number)

Output ONLY the category name (left_right, count, distance, or mcq). No explanation.

Category:"""
        
        try:
            response = self.llm_client.send_message(classification_prompt, is_classification=True)
            category = response.strip().lower().replace('-', '_')
            
            # 驗證分類結果（僅 4 種有效類別）
            valid_categories = ['left_right', 'count', 'distance', 'mcq']
            if category in valid_categories:
                self._category_cache[question] = category
                self._log_step('llm_category_detected', {
                    'question_preview': question[:100],
                    'detected_category': category,
                    'method': 'llm_pure'
                })
                return category
            else:
                # 嘗試模糊匹配
                for valid_cat in valid_categories:
                    if valid_cat in category or category in valid_cat:
                        self._category_cache[question] = valid_cat
                        self._log_step('llm_category_fuzzy_matched', {
                            'question_preview': question[:100],
                            'response': response,
                            'matched_category': valid_cat
                        })
                        return valid_cat
                
                # 都不匹配，預設為 mcq
                self._log_step('llm_category_invalid', {
                    'question_preview': question[:100],
                    'response': response,
                    'fallback': 'mcq'
                })
                return 'mcq'
        except Exception as e:
            self._log_step('llm_category_error', {
                'question_preview': question[:100],
                'error': str(e),
                'error_type': type(e).__name__,
                'fallback': 'mcq'
            })
            return 'mcq'

    def _log_step(self, step_name, data):
        """記錄步驟"""
        log_entry = {
            'step': step_name,
            'data': data,
            'timestamp': None  # 可以添加時間戳
        }
        self.step_log.append(log_entry)
        if self.verbose:
            print(f"[STEP] {step_name}: {str(data)}")

    def set_masks(self):
        # Train set 格式：conversations[0]['value'] 是問題
        conversation = self.input['conversations'][0]['value']
        rle_data = self.input['rle']
        
        self._log_step('set_masks_start', {
            'item_id': self.input.get('id', 'unknown'),
            'conversation_length': len(conversation),
            'rle_count': len(rle_data),
            'conversation_preview': conversation
        })
        
        self.masks = parse_masks_from_conversation(conversation, rle_data)
        
        self._log_step('set_masks_complete', {
            'masks_found': len(self.masks),
            'mask_keys': list(self.masks.keys()),
            'masks_detail': {k: str(v) for k, v in list(self.masks.items())[:5]}
        })
        
        self.tools_api.update_masks(self.masks)
        if not self.masks:
            raise ValueError("No valid masks found in the conversation.")
        
        # 設置原始圖像路徑（根據數據集動態設置）
        # 從 input_item 中獲取圖像路徑，如果沒有則使用默認路徑
        if 'image_dir' in self.input:
            self.original_image_path = os.path.join(self.input['image_dir'], self.input['image'])
        else:
            default_image_dir = '/home/ymtuan/dl/DLCV_1141_final_challenge_1/DLCV_Final1/train/images'
            self.original_image_path = os.path.join(default_image_dir, self.input['image'])
        
        if not os.path.exists(self.original_image_path):
            self._log_step('image_path_not_found', {
                'resolved_path': self.original_image_path
            })
            raise FileNotFoundError(f"Image not found: {self.original_image_path}")
        # 已註解：SoM 功能暫時不使用
        # # 如果有 SoM 可視化器，生成 SoM 圖像
        # if self.som_visualizer:
        #     self._generate_som_image()
    
    # 已註解：SoM 功能暫時不使用
    # def _generate_som_image(self):
    #     """生成 SoM 可視化圖像"""
    #     try:
    #         import os
    #         if not os.path.exists(self.original_image_path):
    #             self._log_step('som_image_skipped_missing_image', {
    #                 'image_path': self.original_image_path
    #             })
    #             self.som_image_path = None
    #             return
    #         som_dir = os.path.join(self.output_dir, 'som_images')
    #         os.makedirs(som_dir, exist_ok=True)
    #         
    #         self.som_image_path = os.path.join(som_dir, f"som_{self.input['id']}.png")
    #         
    #         self.som_visualizer.create_som_visualization(
    #             image_path=self.original_image_path,
    #             masks=self.masks,
    #             output_path=self.som_image_path,
    #             numbering_scheme="region_id"
    #         )
    #         
    #         self._log_step('som_image_generated', {
    #             'som_image_path': self.som_image_path
    #         })
    #         
    #     except Exception as e:
    #         if self.verbose:
    #             print(f"SoM 圖像生成失敗: {e}")
    #         self.som_image_path = self.original_image_path
    
    def _preprocess_question(self, question):
        """預處理問題，將 <mask> 替換為具體的物件代碼"""
        import re
        
        # 找到所有 <mask> 標記
        mask_pattern = re.compile(r'<mask>')
        mask_positions = list(mask_pattern.finditer(question))
        
        if not mask_positions:
            return question
        
        # 按 region_id 排序的 masks
        sorted_masks = sorted(self.masks.items(), key=lambda x: x[1].region_id)
        
        # 從後往前替換，避免位置偏移
        processed_question = question
        for i, match in enumerate(reversed(mask_positions)):
            if i < len(sorted_masks):
                mask_name, mask_obj = sorted_masks[len(mask_positions) - 1 - i]
                replacement = f"{mask_obj.object_class}_{mask_obj.object_id}"
                start, end = match.span()
                processed_question = processed_question[:start] + replacement + processed_question[end:]
        
        self._log_step('question_preprocessed', {
            'original': question,
            'processed': processed_question
        })
        
        return processed_question
    
    def _is_final_tool_result(self, command, result):
        """判斷工具結果是否可以直接作為最終答案"""
        # 解析函數名
        import re
        func_match = re.match(r"(\w+)\s*\(", command.strip())
        if not func_match:
            return False
        
        func_name = func_match.group(1)
        
        # 檢查問題類型來決定是否是最終答案
        question = self.question.lower()
        
        # 如果問題問的是數量/計數，mask名稱通常不是最終答案
        if any(keyword in question for keyword in ['number', 'count', 'how many', 'total']):
            # 只有數值型結果才可能是最終答案
            try:
                float(str(result))
                return func_name in ['dist', 'inside']  # 只有這些函數返回數值
            except:
                return False
        
        # 如果問題問的是"which"、"what"且期望物件名稱，則可以直接返回
        if any(keyword in question for keyword in ['which', 'what']) and not any(keyword in question for keyword in ['number', 'count', 'many']):
            return func_name in ['closest', 'most_right', 'most_left', 'middle', 'is_empty']
        
        # 距離查詢總是可以直接返回數值結果
        if func_name == 'dist':
            return True
        
        # 方向查詢（is_left/is_right）需要特殊處理
        if func_name in ['is_left', 'is_right']:
            # 檢查是否是問方向的問題
            if any(keyword in question for keyword in ['left', 'right', 'side', 'which']):
                return True
            
        return False
    
    def _convert_boolean_to_direction(self, command, boolean_result, reference_question=None):
        """將 Boolean 結果轉換為方向字符串
        
        Args:
            command: 執行的命令
            boolean_result: Boolean 結果
            reference_question: 參考問題文本，用於額外的上下文判斷
        """
        import re
        func_match = re.match(r"(\w+)\s*\(", command.strip())
        if not func_match:
            return str(boolean_result)
        
        func_name = func_match.group(1)
        
        if func_name == 'is_left':
            result = 'left' if boolean_result else 'right'
        elif func_name == 'is_right':
            result = 'right' if boolean_result else 'left'
        else:
            result = str(boolean_result)
        
        self._log_step('boolean_to_direction_converted', {
            'function': func_name,
            'boolean_result': boolean_result,
            'direction_result': result,
            'reference_question': reference_question
        })
        return result
    
    def _is_direction_question(self, command, result):
        """檢查是否是方向問題"""
        import re
        func_match = re.match(r"(\w+)\s*\(", command.strip())
        if not func_match:
            return False
        
        func_name = func_match.group(1)
        return func_name in ['is_left', 'is_right'] and isinstance(result, bool)
    
    def format_answer(self, raw_answer=None):
        """格式化答案，可以接受來自 conversation_loop 的答案或重新查詢 LLM"""
        
        if raw_answer is not None:
            # 使用提供的答案（來自 conversation_loop）
            answer = raw_answer.strip()
            
            self._log_step('format_answer_from_conversation', {
                'raw_answer': answer,
                'question_category': self.question_category
            })
        else:
            # 重新查詢 LLM（舊行為）
            self._log_step('format_answer_start', {
                'answer_preamble': self.answer_preamble[:100]
            })
            
            answer = self._send_with_retry(self.answer_preamble)
            answer = answer.strip()
            
            self._log_step('format_answer_response', {
                'raw_answer': answer
            })
        
        # 使用類別特定的答案提取
        formatted_answer = self._extract_final_answer(answer, category=self.question_category)
        
        # 如果提取失敗（返回 None），說明 LLM 說無法計算
        if formatted_answer is None:
            self._log_step('format_answer_cannot_extract', {
                'raw_answer': answer[:200],
                'category': self.question_category,
                'reason': 'LLM indicated cannot compute, returning error marker'
            })
            # 返回錯誤標記，表示需要多步驟計算
            return "-1"  # 使用 -1 作為錯誤標記
        
        self._log_step('format_answer_extracted', {
            'formatted_answer': formatted_answer,
            'category': self.question_category
        })
        
        # 如果是左右問題且答案是 true/false，轉換為方向
        if self.question_category == 'left_right' and formatted_answer.lower() in ['true', 'false']:
            # 需要從問題或最後的工具命令推斷
            formatted_answer = self._convert_boolean_answer_to_direction(formatted_answer)
            self._log_step('format_answer_boolean_converted', {
                'original': formatted_answer,
                'converted': formatted_answer
            })
        
        # 如果答案是 mask 名稱，需要根據問題類型決定如何處理
        if formatted_answer in self.masks:
            mask_obj = self.masks[formatted_answer]
            # 檢查原問題來決定返回格式
            original_question = self.input['conversations'][0]['value'].lower()
            
            # 檢查標準答案格式，如果標準答案是 Region ID，則返回 Region ID
            gt_answer = self.input.get('normalized_answer', '')
            if gt_answer and gt_answer.isdigit():
                # 標準答案是數字，可能是 Region ID 或 object_id
                # 檢查標準答案是否匹配 Region ID
                if str(mask_obj.region_id) == gt_answer:
                    # 標準答案匹配 Region ID，返回 Region ID
                    result = str(mask_obj.region_id)
                    self._log_step('format_answer_mapped_to_region_id', {
                        'original': formatted_answer,
                        'mapped_to': result,
                        'reason': 'GT answer matches region_id, returning region_id'
                    })
                    return result
                elif str(mask_obj.object_id) == gt_answer:
                    # 標準答案匹配 object_id，返回 object_id
                    result = str(mask_obj.object_id)
                    self._log_step('format_answer_mapped_to_object_id', {
                        'original': formatted_answer,
                        'mapped_to': result,
                        'reason': 'GT answer matches object_id, returning object_id'
                    })
                    return result
            
            # 如果無法從 GT 判斷，使用原有邏輯
            if any(word in original_question for word in ['which', 'what', 'who']):
                # 對於 "which" 類型問題，優先返回 Region ID（與數據集格式一致）
                # 但如果問題明確提到 object_id，則返回 object_id
                result = str(mask_obj.region_id)
                self._log_step('format_answer_mapped_to_region_id', {
                    'original': formatted_answer,
                    'mapped_to': result,
                    'reason': 'Which-type question, using region_id to match dataset format'
                })
            else:
                # 對於其他問題，返回 Region ID
                result = str(mask_obj.region_id)
                self._log_step('format_answer_mapped_to_region_id', {
                    'original': formatted_answer,
                    'mapped_to': result,
                    'reason': 'Non-which question, using region_id'
                })
            
            return result
        else:
            self._log_step('format_answer_direct', {
                'answer': formatted_answer
            })
            return formatted_answer
    
    def _convert_boolean_answer_to_direction(self, boolean_str):
        """根據問題和可能的工具命令，將布爾字符串轉換為方向
        
        This is a fallback when we have boolean answer but need direction
        """
        question_lower = self.question.lower()
        boolean_lower = boolean_str.lower()
        
        # 檢查問題中提到的方向
        # 如果問題問 "Is X to the left of Y?" 且答案是 true，則答案是 left
        # 如果問題問 "Is X to the right of Y?" 且答案是 true，則答案是 right
        
        if 'is_left' in question_lower or 'to the left' in question_lower:
            return 'left' if boolean_lower == 'true' else 'right'
        elif 'is_right' in question_lower or 'to the right' in question_lower:
            return 'right' if boolean_lower == 'true' else 'left'
        else:
            # 預設：true -> left, false -> right (based on is_left函數邏輯)
            return 'left' if boolean_lower == 'true' else 'right'

    def _extract_final_answer(self, answer_text, category=None):
        """從答案文本中提取最終答案（類別特定）"""
        import re
        
        self._log_step('extract_final_answer_start', {
            'answer_text': answer_text[:200],
            'category': category
        })
        
        # 左右問題：優先尋找方向關鍵詞，其次 Boolean 值
        if category == 'left_right':
            # 1. 優先尋找明確的方向關鍵詞
            direction_match = re.search(r'\b(left|right)\b', answer_text.lower())
            if direction_match:
                result = direction_match.group(1)
                self._log_step('extract_final_answer_left_right', {
                    'result': result,
                    'reason': 'Direction keyword found'
                })
                return result
            
            # 2. 尋找 Boolean 值（如果沒找到方向關鍵詞）
            # 但不要直接返回 true/false，因為稍後會被轉換
            bool_match = re.search(r'\b(true|false|yes|no)\b', answer_text.lower())
            if bool_match:
                bool_str = bool_match.group(1).lower()
                self._log_step('extract_final_answer_left_right_bool', {
                    'bool_value': bool_str,
                    'reason': 'Boolean value found, will be converted to direction'
                })
                # 直接轉換 boolean 為方向而不是返回 true/false 字符串
                return 'left' if bool_str in ['true', 'yes'] else 'right'
            
            return answer_text
        
        # 計數問題：尋找數字
        if category == 'count':
            numbers = re.findall(r'\b(\d+)\b', answer_text)
            if numbers:
                result = numbers[-1]  # 取最後一個數字
                self._log_step('extract_final_answer_count', {
                    'result': result,
                    'all_numbers': numbers
                })
                return result
        
        # 距離問題：尋找浮點数
        if category == 'distance':
            distance_patterns = [
                r'(?:distance|dist).*?is\s+(\d+\.?\d*)',
                r'(\d+\.?\d*)\s*(?:units?|meters?|m\b)',
                r'(?:is|=)\s*(\d+\.?\d*)',
            ]
            
            for pattern in distance_patterns:
                match = re.search(pattern, answer_text, re.IGNORECASE)
                if match:
                    result = match.group(1)
                    self._log_step('extract_final_answer_distance', {
                        'result': result,
                        'pattern': pattern
                    })
                    return result
        
        # MCQ 問題：尋找物件名稱或 object_id
        if category == 'mcq':
            # 優先尋找 "Region X" 格式（這是數據集中使用的格式）
            region_pattern = r'\[Region\s+(\d+)\]'
            region_matches = re.findall(region_pattern, answer_text, re.IGNORECASE)
            if region_matches:
                # 找到最後一個提到的 Region（通常是答案）
                region_id = int(region_matches[-1])
                # 建立 region_id 到 mask 的映射
                region_to_mask = {mask_obj.region_id: mask_name 
                                 for mask_name, mask_obj in self.masks.items()}
                
                if region_id in region_to_mask:
                    mask_name = region_to_mask[region_id]
                    mask_obj = self.masks[mask_name]
                    # 返回 Region ID（與標準答案格式一致）
                    result = str(region_id)
                    self._log_step('extract_final_answer_mcq_region', {
                        'region_id': region_id,
                        'mask_name': mask_name,
                        'object_id': mask_obj.object_id,
                        'result': result,
                        'reason': 'Found Region X format, returning region_id to match normalized_answer format'
                    })
                    return result
            
            # 優先尋找 mask 名稱（buffer_, pallet_, transporter_, shelf_）
            mask_match = re.search(r'\b(buffer_\d+|pallet_\d+|transporter_\d+|shelf_\d+|object_\d+)\b', answer_text)
            if mask_match:
                result = mask_match.group(1)
                self._log_step('extract_final_answer_mcq_mask', {
                    'result': result
                })
                return result
            
            # 尋找單獨的數字（可能是 object_id 或 region_id）
            numbers = re.findall(r'\b(\d+)\b', answer_text)
            if numbers:
                result = numbers[-1]
                self._log_step('extract_final_answer_mcq_number', {
                    'result': result,
                    'all_numbers': numbers
                })
                return result
        
        # 通用後備邏輯
        # 先嘗試尋找距離相關的數字模式
        distance_patterns = [
            r'(?:distance|dist).*?is\s+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*(?:units?|meters?|m\b)',
            r'(?:is|=)\s*(\d+\.?\d*)',
        ]
        
        for pattern in distance_patterns:
            match = re.search(pattern, answer_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # 尋找方向關鍵詞
        direction_match = re.search(r'\b(left|right|center|middle)\b', answer_text.lower())
        if direction_match:
            return direction_match.group(1)
        
        # 尋找 mask 名稱
        mask_match = re.search(r'\b(buffer_\d+|pallet_\d+|transporter_\d+|shelf_\d+|object_\d+)\b', answer_text)
        if mask_match:
            return mask_match.group(1)
        
        # 最後才尋找任意數字
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', answer_text)
        if numbers:
            filtered_numbers = []
            for num in numbers:
                num_pos = answer_text.find(num)
                if num_pos > 0 and answer_text[num_pos-1] == '_':
                    continue
                filtered_numbers.append(num)
            
            if filtered_numbers:
                return filtered_numbers[-1]
        
        # 檢測是否為"無法計算"類型的回答
        cannot_compute_patterns = [
            r'cannot.*comput',
            r'cannot.*calculat',
            r'unable.*comput',
            r'unable.*calculat',
            r'cannot.*one.*function',
            r'need.*multiple.*step',
            r'requires.*multiple.*function'
        ]
        
        answer_lower = answer_text.lower()
        for pattern in cannot_compute_patterns:
            if re.search(pattern, answer_lower):
                self._log_step('extract_final_answer_cannot_compute', {
                    'answer_text': answer_text[:200],
                    'pattern': pattern,
                    'reason': 'LLM says cannot compute, this should trigger multi-step guidance'
                })
                # 返回特殊標記，讓 format_answer 知道需要處理
                return None  # 返回 None 表示無法提取有效答案
        
        # 如果都找不到，返回原答案
        return answer_text

    def set_question(self):
        self.messages = []
        original_question = self.input['conversations'][0]['value']
        self.question = self._preprocess_question(original_question)
        
        # 保存 ground truth category（用於後續準確率計算）
        self.ground_truth_category = self.input.get('category')
        
        # 根據 force_llm_category 決定是否使用 LLM 分類
        if self.force_llm_category:
            # 強制使用 LLM 分類，跳過數據集 category 和關鍵字檢測
            self.predicted_category = self._detect_question_category_with_llm(self.question)
            self.question_category = self.predicted_category
            self._log_step('category_forced_llm', {
                'category': self.question_category,
                'ground_truth_category': self.ground_truth_category,
                'source': 'llm_forced'
            })
        else:
            # 原有邏輯：優先使用資料集中提供的 category
            if 'category' in self.input and self.input['category']:
                self.question_category = self.input['category']
                self.predicted_category = self.question_category
                self._log_step('category_from_dataset', {
                    'category': self.question_category,
                    'source': 'dataset'
                })
            else:
                self.question_category = self._detect_question_category_with_llm(self.question)
                self.predicted_category = self.question_category
                self._log_step('category_from_llm_fallback', {
                    'category': self.question_category,
                    'source': 'llm',
                    'reason': 'keyword_detection_returned_other'
                })
        
        self._log_step('set_question_start', {
            'original_question': original_question,
            'processed_question': self.question,
            'predicted_category': self.predicted_category,
            'ground_truth_category': self.ground_truth_category,
            'image': self.input['image']
        })
        
        full_prompt = self.prompt_preamble.replace("<question>", self.question)
        
        # 加入可用 masks 資訊到 prompt
        available_masks = ", ".join(self.masks.keys())
        full_prompt += f"\n\nAvailable masks in this image: {available_masks}"
        
        # 使用原始圖像給 VLM（已註解 SoM 功能）
        # image_for_vlm = self.som_image_path if self.som_image_path else self.original_image_path
        image_for_vlm = self.original_image_path
        
        self._log_step('prompt_prepared', {
            'prompt_length': len(full_prompt),
            'available_masks': available_masks,
            'vlm_image_path': image_for_vlm,
            'original_image_path': self.original_image_path,
            'question_category': self.question_category
        })
        
        self.messages.append({"role": "user", "content": full_prompt})
        
        # 重要：tools_api 仍使用原始圖像進行距離等計算
        self.tools_api.update_image(self.original_image_path)
        
        # VLM 客戶端使用 SoM 圖像
        if hasattr(self.llm_client, 'update_image'):
            self.llm_client.update_image(image_for_vlm)
        
        return self._conversation_loop()

    def _send_with_retry(self, message, retries=3, backoff=1.0):
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                return self.llm_client.send_message(message)
            except Exception as exc:
                last_exc = exc
                self._log_step('llm_send_retry_failed', {
                    'attempt': attempt,
                    'retries': retries,
                    'error': str(exc),
                    'error_type': type(exc).__name__
                })
                time.sleep(backoff * attempt)
        raise last_exc

    def _conversation_loop(self, budget=10):
        usage = 0
        execute_flag = False
        last_tool_result = None
        last_tool_command = None
        self.llm_client.reset_conversation()
        
        self._log_step('conversation_loop_start', {
            'budget': budget,
            'initial_message_count': len(self.messages),
            'question_category': self.question_category
        })
        
        while usage < budget:
            usage += 1
            
            self._log_step(f'iteration_{usage}_start', {
                'iteration': usage,
                'message_count': len(self.messages),
                'execute_flag': execute_flag
            })
            
            try:
                latest_message = self.messages[-1]["content"]
                
                self._log_step(f'iteration_{usage}_send_to_llm', {
                    'message_length': len(latest_message),
                    'message_preview': latest_message[:200]
                })
                
                response_text = self._send_with_retry(latest_message)
                
                self._log_step(f'iteration_{usage}_llm_response', {
                    'response_length': len(response_text),
                    'response_preview': response_text[:200]
                })
                
                self.messages = self.llm_client.messages.copy()

                # 檢查是否有 <final> tag（LLM 標記工具結果為最終答案）
                # 先嘗試匹配完整的 <final>...</final>
                final_match = re.search(r"<final>(.*?)</final>", response_text, re.IGNORECASE | re.DOTALL)
                
                # 如果沒有完整標籤，嘗試匹配不完整的 <final>...（沒有閉合標籤）
                if not final_match:
                    incomplete_final_match = re.search(r"<final>\s*(\S+.*?)$", response_text, re.IGNORECASE | re.DOTALL)
                    if incomplete_final_match:
                        final_content = incomplete_final_match.group(1).strip()
                        self._log_step('incomplete_final_tag_detected', {
                            'final_content': final_content,
                            'reason': 'LLM provided <final> without closing tag, extracted content anyway'
                        })
                        final_match = incomplete_final_match  # 使用這個匹配結果
                
                if final_match:
                    final_content = final_match.group(1).strip()
                    
                    # 如果 <final> 內有內容，直接使用該內容
                    if final_content:
                        self._log_step('final_tag_with_content', {
                            'final_content': final_content,
                            'reason': 'LLM provided explicit answer in <final> tag'
                        })
                        return final_content
                    
                    # 如果 <final> 為空但有工具結果，使用工具結果
                    elif last_tool_result is not None:
                        # 特殊處理方向函數的 Boolean 結果
                        if self._is_direction_question(last_tool_command, last_tool_result):
                            direction_result = self._convert_boolean_to_direction(
                                last_tool_command, 
                                last_tool_result,
                                reference_question=self.question
                            )
                            self._log_step('direction_result_converted_at_final', {
                                'command': last_tool_command,
                                'boolean_result': str(last_tool_result),
                                'direction_result': direction_result,
                                'reason': 'Empty <final> tag, converted boolean to direction string',
                                'question_category': self.question_category
                            })
                            return direction_result
                        
                        self._log_step('final_tool_result', {
                            'command': last_tool_command,
                            'result': str(last_tool_result),
                            'result_type': type(last_tool_result).__name__,
                            'reason': 'Empty <final> tag, using tool result as final answer',
                            'question_category': self.question_category
                        })
                        return str(last_tool_result)

                execute_match = re.search(r"<execute>(.*?)</execute>", response_text, re.DOTALL)
                
                if not execute_match:
                    incomplete_execute = re.search(r"<execute>\s*$", response_text, re.DOTALL)
                    if incomplete_execute:
                        self._log_step(f'iteration_{usage}_incomplete_execute', {
                            'response_preview': response_text[-100:]
                        })
                        
                        self.messages.append({
                            "role": "user", 
                            "content": "Please complete your execute command. Write the specific function call inside <execute>...</execute> tags."
                        })
                        continue
                
                if execute_match:
                    execute_flag = True
                    command = execute_match.group(1).strip()
                    
                    self._log_step(f'iteration_{usage}_execute_found', {
                        'command': command
                    })
                    
                    # 檢測是否應該先調用空間函數但沒有調用
                    question_lower = self.question.lower()
                    reasoning_text = response_text.lower()
                    
                    # 檢查問題中是否提到 leftmost/rightmost/middle
                    question_mentions_leftmost = any(kw in question_lower for kw in ['leftmost', 'most left', 'furthest left'])
                    question_mentions_rightmost = any(kw in question_lower for kw in ['rightmost', 'most right', 'furthest right'])
                    question_mentions_middle = 'middle' in question_lower
                    
                    # 檢查當前命令是否調用了空間函數
                    command_calls_spatial = any(func in command for func in ['most_left', 'most_right', 'middle'])
                    
                    # 檢查對話歷史中是否已經調用過空間函數（避免誤判）
                    # 檢查之前的工具結果，看是否已經找到過 leftmost/rightmost/middle
                    has_called_spatial_before = False
                    if last_tool_command:
                        has_called_spatial_before = any(func in last_tool_command for func in ['most_left', 'most_right', 'middle'])
                    
                    # 檢查所有歷史消息，看是否調用過空間函數
                    for msg in self.messages:
                        if isinstance(msg, dict) and 'content' in msg:
                            content = msg['content']
                            if any(func in content for func in ['most_left', 'most_right', 'middle']):
                                has_called_spatial_before = True
                                break
                    
                    # 如果問題提到空間關係但命令沒有調用空間函數，且命令中使用了具體的 mask 名稱
                    # 並且之前沒有調用過空間函數（避免誤判已經找到的結果）
                    if (question_mentions_leftmost or question_mentions_rightmost or question_mentions_middle) and not command_calls_spatial and not has_called_spatial_before:
                        # 檢查命令中是否直接使用了 mask 名稱（可能是假設的）
                        mask_name_pattern = r'\b(pallet|buffer|transporter|shelf)_\d+'
                        if re.search(mask_name_pattern, command):
                            # 提醒 LLM 應該先調用空間函數
                            if question_mentions_leftmost:
                                self.messages.append({
                                    "role": "user",
                                    "content": "WARNING: The question asks about 'leftmost', but you're using a specific mask name directly. You MUST first call most_left([list_of_masks]) to find the leftmost object, then use that result in your next function call."
                                })
                                continue
                            elif question_mentions_rightmost:
                                self.messages.append({
                                    "role": "user",
                                    "content": "WARNING: The question asks about 'rightmost', but you're using a specific mask name directly. You MUST first call most_right([list_of_masks]) to find the rightmost object, then use that result in your next function call."
                                })
                                continue
                            elif question_mentions_middle:
                                self.messages.append({
                                    "role": "user",
                                    "content": "WARNING: The question asks about 'middle', but you're using a specific mask name directly. You MUST first call middle([list_of_masks]) to find the middle object, then use that result in your next function call."
                                })
                                continue
                    
                    try:
                        result = self._execute_function(command)
                        
                        last_tool_result = result
                        last_tool_command = command
                        
                        self._log_step(f'iteration_{usage}_execute_result', {
                            'result': str(result),
                            'result_type': type(result).__name__,
                            'question_category': self.question_category
                        })
                        
                        self.messages.append({"role": "user", "content": f"Tool result: {result}"})
                        continue
                    except Exception as exec_exc:
                        self._log_step(f'iteration_{usage}_execute_error', {
                            'command': command,
                            'error': str(exec_exc),
                            'error_type': type(exec_exc).__name__
                        })
                        
                        # 嘗試提示 LLM 重試或使用其他方法
                        self.messages.append({
                            "role": "user",
                            "content": f"Error executing function: {str(exec_exc)}. Please try a different approach or function."
                        })
                        continue

                answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
                if not answer_match:
                    answer_start_match = re.search(r"<answer>\s*(.*?)$", response_text, re.DOTALL | re.MULTILINE)
                    if answer_start_match:
                        answer_match = answer_start_match
                        
                if answer_match and execute_flag:
                    final_answer = answer_match.group(1).strip()
                    
                    # 檢測是否為"無法計算"類型的回答
                    cannot_compute_keywords = ['cannot', 'unable', 'need multiple', 'requires multiple', 'one function call']
                    if any(keyword in final_answer.lower() for keyword in cannot_compute_keywords):
                        self._log_step('final_answer_cannot_compute_detected', {
                            'answer': final_answer,
                            'reason': 'LLM says cannot compute, guiding to use multi-step approach'
                        })
                        # 引導 LLM 使用多步驟方法
                        question_lower = self.question.lower()
                        guidance_msg = "You can solve this in multiple steps! "
                        
                        if 'rightmost' in question_lower or 'most right' in question_lower:
                            guidance_msg += "First, use most_right([list_of_objects]) to find the rightmost object. "
                        if 'leftmost' in question_lower or 'most left' in question_lower:
                            guidance_msg += "Use most_left([list_of_objects]) to find the leftmost object. "
                        if 'distance' in question_lower:
                            guidance_msg += "Then use dist(object1, object2) to calculate the distance. "
                        if 'closest' in question_lower:
                            guidance_msg += "Then use closest(object1, [list_of_objects]) to find the closest. "
                        
                        guidance_msg += "Please try again with multiple function calls."
                        
                        self.messages.append({
                            "role": "user",
                            "content": guidance_msg
                        })
                        continue
                    
                    self._log_step('final_answer_found', {
                        'answer': final_answer,
                        'total_iterations': usage,
                        'category': self.question_category
                    })
                    
                    if self.question_category == 'left_right':
                        final_answer_lower = final_answer.lower()
                        if final_answer_lower in ['true', 'false', 'yes', 'no']:
                            final_answer = 'left' if final_answer_lower in ['true', 'yes'] else 'right'
                            self._log_step('final_answer_boolean_converted', {
                                'original': final_answer_lower,
                                'converted': final_answer
                            })
                    
                    self.messages.append({"role": "assistant", "content": final_answer})
                    return final_answer

                self._log_step(f'iteration_{usage}_no_valid_action', {
                    'response_preview': response_text[:300],
                    'has_execute': execute_match is not None,
                    'has_answer': answer_match is not None
                })
                
                # 檢測是否在 reasoning 中提到了 leftmost/rightmost/middle 但沒有調用工具
                reasoning_text = response_text.lower()
                needs_spatial_tool = False
                tool_suggestion = ""
                
                if any(keyword in reasoning_text for keyword in ['leftmost', 'rightmost', 'furthest left', 'furthest right', 'most left', 'most right']):
                    if 'most_left' not in response_text and 'most_right' not in response_text:
                        needs_spatial_tool = True
                        if 'leftmost' in reasoning_text or 'most left' in reasoning_text or 'furthest left' in reasoning_text:
                            tool_suggestion = "You mentioned 'leftmost' but didn't call most_left(). Please call most_left([list_of_masks]) first to find the leftmost object."
                        else:
                            tool_suggestion = "You mentioned 'rightmost' but didn't call most_right(). Please call most_right([list_of_masks]) first to find the rightmost object."
                
                if 'middle' in reasoning_text and 'middle(' not in response_text:
                    needs_spatial_tool = True
                    tool_suggestion = "You mentioned 'middle' but didn't call middle(). Please call middle([mask1, mask2, mask3]) to find the middle object."
                
                if needs_spatial_tool:
                    self.messages.append({
                        "role": "user",
                        "content": f"IMPORTANT: {tool_suggestion} You MUST use the spatial functions (most_left, most_right, middle) to determine object positions. Do not assume based on ID numbers."
                    })
                else:
                    # 提示 LLM 提供有效的結構
                    self.messages.append({
                        "role": "user",
                        "content": "Please provide a clear response in one of these formats:\n1. <execute>function_name(args)</execute> to call a tool\n2. <answer>your_answer</answer> after using tools"
                    })
                
            except Exception as loop_exc:
                self._log_step(f'iteration_{usage}_loop_error', {
                    'error': str(loop_exc),
                    'error_type': type(loop_exc).__name__,
                    'traceback': f"{loop_exc.__class__.__name__}"
                })
                
                if usage >= budget:
                    raise ValueError(f"Conversation loop exceeded budget. Last error: {str(loop_exc)}")
                else:
                    # 繼續下一次迭代
                    continue
        
        raise ValueError(f"Failed to get answer after {budget} iterations")

    def _execute_function(self, command):
        original_command = command
        
        try:
            lines = [line.strip() for line in command.split('\n') if line.strip()]
            
            self._log_step('parse_function_call', {
                'original_command': original_command,
                'command_lines': lines,
                'line_count': len(lines)
            })
            
            if len(lines) > 1:
                return self._execute_multiline_command(lines)
            else:
                return self._execute_single_command(lines[0] if lines else command)
        except Exception as e:
            self._log_step('execute_function_error', {
                'command': original_command,
                'error': str(e),
                'error_type': type(e).__name__
            })
            raise

    def _execute_multiline_command(self, lines):
        """執行多行命令並保存中間結果"""
        variables = {}
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            try:
                assignment_match = re.match(r'(\w+)\s*=\s*(.+)', line)
                if assignment_match:
                    var_name = assignment_match.group(1)
                    func_call = assignment_match.group(2).strip()
                    
                    result = self._execute_single_command(func_call, variables)
                    variables[var_name] = result
                    
                    self._log_step(f'multiline_step_{i}', {
                        'variable': var_name,
                        'function_call': func_call,
                        'result': str(result),
                        'result_type': type(result).__name__
                    })
                else:
                    result = self._execute_single_command(line, variables)
                    self._log_step('multiline_final_result', {
                        'command': line,
                        'result': str(result),
                        'result_type': type(result).__name__
                    })
                    return result
            except Exception as e:
                self._log_step(f'multiline_step_{i}_error', {
                    'line': line,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'variables_available': list(variables.keys())
                })
                raise
        
        if variables:
            last_var = list(variables.keys())[-1]
            return variables[last_var]
        
        return None

    def _execute_single_command(self, command, variables=None):
        """執行單個命令"""
        if variables is None:
            variables = {}
        
        try:
            command = command.strip().replace('\\n', '').replace('\n', '').replace('\r', '').replace('<', '').replace('>', '')
            
            if not command:
                raise ValueError("Empty command")
            
            self._log_step('execute_single_command', {
                'command': command,
                'available_variables': list(variables.keys()),
                'available_masks': list(self.masks.keys()) if self.masks else []
            })
            
            match = re.match(r"(\w+)\s*\((.*)\)", command)
            if not match:
                raise ValueError(f"Invalid function call format: {command}")

            func_name, args_str = match.groups()
            
            self._log_step('function_identified', {
                'function_name': func_name,
                'args_string': args_str
            })
            
            func_map = {
                "dist": self.tools_api.dist,
                "closest": self.tools_api.closest,
                "is_left": self.tools_api.is_left,
                "is_right": self.tools_api.is_right,
                "inside": self.tools_api.inside,
                "most_right": self.tools_api.most_right,
                "most_left": self.tools_api.most_left,
                "middle": self.tools_api.middle,
                "is_empty": self.tools_api.is_empty
            }

            if func_name not in func_map:
                raise ValueError(f"Unknown function: {func_name}. Available: {list(func_map.keys())}")

            try:
                parsed_args = self._parse_arguments(args_str, variables)
            except Exception as parse_exc:
                self._log_step('argument_parsing_error', {
                    'args_str': args_str,
                    'error': str(parse_exc),
                    'variables': list(variables.keys()),
                    'masks': list(self.masks.keys()) if self.masks else []
                })
                raise
            
            self._log_step('arguments_parsed', {
                'parsed_args': [str(arg)[:50] for arg in parsed_args],
                'arg_types': [type(arg).__name__ for arg in parsed_args],
                'arg_count': len(parsed_args)
            })
            
            result = func_map[func_name](*parsed_args)
            
            self._log_step('function_executed', {
                'function': func_name,
                'result': str(result)[:100],
                'result_type': type(result).__name__
            })
            
            return result
        except Exception as e:
            self._log_step('single_command_error', {
                'command': command,
                'error': str(e),
                'error_type': type(e).__name__
            })
            raise

    def _parse_arguments(self, args_str, variables=None):
        if variables is None:
            variables = {}
        
        try:
            def resolve(node):
                if isinstance(node, ast.Name):
                    key = node.id
                    
                    if key in variables:
                        return variables[key]
                    
                    if key.startswith('<') and key.endswith('>'):
                        key = key[1:-1]
                    
                    if key not in self.masks:
                        raise ValueError(f"Mask '{key}' not found. Available masks: {list(self.masks.keys())}")
                    
                    return self.masks[key]
                elif isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.List):
                    return [resolve(elt) for elt in node.elts]
                elif isinstance(node, ast.Tuple):
                    return tuple(resolve(elt) for elt in node.elts)
                elif isinstance(node, ast.Str):
                    return node.s
                else:
                    raise ValueError(f"Unsupported AST node: {ast.dump(node)}")

            tree = ast.parse(f"func({args_str})", mode='eval')
            func_call = tree.body
            resolved_args = [resolve(arg) for arg in func_call.args]
            return tuple(resolved_args)
        except Exception as e:
            self._log_step('parse_arguments_error', {
                'args_str': args_str,
                'error': str(e),
                'error_type': type(e).__name__
            })
            raise

def normalize_answer(pred: str, gt: str, tolerance_percent: float = 10.0) -> bool:
    """比較答案是否正確（處理不同格式）
    
    Args:
        pred: 預測答案
        gt: 真實答案  
        tolerance_percent: 數值答案的相對容錯百分比（預設 10%）
    """
    pred = str(pred).strip().lower()
    gt = str(gt).strip().lower()
    
    # 直接比較
    if pred == gt:
        return True
    
    # 處理數字答案（距離、計數等）
    try:
        pred_num = float(pred)
        gt_num = float(gt)
        
        # 如果真實值為 0，使用絕對誤差
        if gt_num == 0:
            return abs(pred_num - gt_num) < 0.1
        
        # 使用相對誤差（acc@tolerance_percent）
        relative_error = abs(pred_num - gt_num) / abs(gt_num) * 100
        if relative_error <= tolerance_percent:
            return True
            
        # 備用：絕對誤差（針對非常小的數值）
        if abs(pred_num - gt_num) < 0.1:
            return True
            
    except:
        pass
    
    # 處理 left/right
    if pred in ['left', 'right'] and gt in ['left', 'right']:
        return pred == gt
    
    return False

def analyze_errors(results: list, output_dir: str = '../output', dataset: str = 'train'):
    """分析錯誤案例"""
    errors = [r for r in results if not r.get('correct', False)]
    
    if not errors:
        print(f"\n沒有錯誤案例（{dataset} set）")
        return []
    
    print(f"\n錯誤分析 ({dataset} set):")
    print(f"總錯誤數: {len(errors)}")
    
    # 按類別分析
    error_by_category = defaultdict(list)
    for err in errors:
        category = err.get('category', 'unknown')
        error_by_category[category].append(err)
    
    print(f"\n按類別分布:")
    for cat, errs in error_by_category.items():
        print(f"  {cat}: {len(errs)} 個錯誤")
    
    # 常見錯誤模式
    error_patterns = Counter()
    for err in errors:
        pred = str(err.get('predicted', '')).lower()
        gt = str(err.get('ground_truth', '')).lower()
        pattern = f"predicted={pred}, gt={gt}"
        error_patterns[pattern] += 1
    
    print(f"\n常見錯誤模式（前10個）:")
    for pattern, count in error_patterns.most_common(10):
        print(f"  {pattern}: {count} 次")
    
    # 詳細錯誤類型分析
    print(f"\n詳細錯誤類型:")
    
    # 方向錯誤
    direction_errors = [e for e in errors if e.get('category') == 'left_right']
    if direction_errors:
        print(f"  方向判斷錯誤: {len(direction_errors)} 個")
        # 檢查是否有明顯的偏向
        left_to_right = sum(1 for e in direction_errors if str(e.get('predicted', '')).lower() == 'left' and str(e.get('ground_truth', '')).lower() == 'right')
        right_to_left = sum(1 for e in direction_errors if str(e.get('predicted', '')).lower() == 'right' and str(e.get('ground_truth', '')).lower() == 'left')
        print(f"    Left→Right 錯誤: {left_to_right} 次")
        print(f"    Right→Left 錯誤: {right_to_left} 次")
        if left_to_right > right_to_left * 1.5:
            print(f"    → 建議: 可能存在左右判斷偏差，檢查 is_left/is_right 工具函數")
        elif right_to_left > left_to_right * 1.5:
            print(f"    → 建議: 可能存在右左判斷偏差，檢查 is_left/is_right 工具函數")
    
    # 計數錯誤
    count_errors = [e for e in errors if e.get('category') == 'count']
    if count_errors:
        print(f"  計數錯誤: {len(count_errors)} 個")
        # 分析計數誤差分布
        count_diffs = []
        for e in count_errors:
            try:
                pred = float(str(e.get('predicted', 0)))
                gt = float(str(e.get('ground_truth', 0)))
                diff = pred - gt
                count_diffs.append(diff)
            except:
                pass
        
        if count_diffs:
            avg_diff = sum(count_diffs) / len(count_diffs)
            abs_avg_diff = sum(abs(d) for d in count_diffs) / len(count_diffs)
            print(f"    平均誤差: {avg_diff:.2f}")
            print(f"    平均絕對誤差: {abs_avg_diff:.2f}")
            
            off_by_one = sum(1 for d in count_diffs if abs(d) == 1)
            off_by_two = sum(1 for d in count_diffs if abs(d) == 2)
            print(f"    差1個: {off_by_one} 次")
            print(f"    差2個: {off_by_two} 次")
            
            if avg_diff > 0.5:
                print(f"    → 建議: 系統傾向高估數量，可能計算了多餘的物體")
            elif avg_diff < -0.5:
                print(f"    → 建議: 系統傾向低估數量，可能遺漏了物體")
    
    # 距離錯誤
    distance_errors = [e for e in errors if e.get('category') == 'distance']

    if distance_errors:
        print(f"  距離估計錯誤: {len(distance_errors)} 個")
        # 分析距離誤差
        dist_relative_errors = []
        for e in distance_errors:
            try:
                pred = float(str(e.get('predicted', 0)))
                gt = float(str(e.get('ground_truth', 0)))
                if gt > 0:
                    rel_error = abs(pred - gt) / gt * 100
                    dist_relative_errors.append(rel_error)
            except:
                pass
        
        if dist_relative_errors:
            avg_rel_error = sum(dist_relative_errors) / len(dist_relative_errors)
            print(f"    平均相對誤差: {avg_rel_error:.1f}%")
            
            if avg_rel_error > 20:
                print(f"    → 建議: 距離估計模型可能需要重新訓練或調整")
    
    # MCQ 錯誤
    mcq_errors = [e for e in errors if e.get('category') == 'mcq']
    if mcq_errors:
        print(f"  選擇題錯誤: {len(mcq_errors)} 個")
        print(f"    → 建議: 檢查物體識別和空間推理邏輯")
    
    # 保存錯誤案例到獨立的 JSON 文件
    error_file = os.path.join(output_dir, f'{dataset}_errors.json')
    with open(error_file, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"\n錯誤案例已保存到: {error_file}")
    
    # 生成詳細的錯誤報告（包含問題和對話歷史）
    detailed_error_file = os.path.join(output_dir, f'{dataset}_errors_detailed.json')
    detailed_errors = []
    for err in errors[:20]:  # 只保存前20個詳細錯誤
        detailed_err = {
            'id': err.get('id'),
            'category': err.get('category'),
            'question': err.get('question'),
            'predicted': err.get('predicted'),
            'ground_truth': err.get('ground_truth'),
            'conversation_summary': [
                {
                    'role': msg.get('role'),
                    'content': str(msg.get('content'))[:200] + '...' if len(str(msg.get('content', ''))) > 200 else str(msg.get('content', ''))
                }
                for msg in err.get('conversation', [])[-5:]  # 只保留最後5輪對話
            ]
        }
        detailed_errors.append(detailed_err)
    
    with open(detailed_error_file, 'w') as f:
        json.dump(detailed_errors, f, indent=2, ensure_ascii=False)
    print(f"詳細錯誤報告已保存到: {detailed_error_file}")
    
    return errors

def main():
    parser = ArgumentParser(description="Evaluate on train/test/val set")
    parser.add_argument('--dataset', type=str, default='train', 
                       choices=['train', 'test', 'val'],
                       help='Dataset to evaluate: train, test, or val')
    parser.add_argument('--project_id', type=str, help='Google Cloud Project ID (for Gemini)')
    parser.add_argument('--location', type=str, default='global', help='Location for Vertex AI')
    parser.add_argument('--think_mode', action='store_true', help='Enable think mode')
    parser.add_argument('--limit', type=int, default=2000, help='Number of samples to process')
    parser.add_argument('--llm_type', type=str, default='vllm', 
                       choices=['gemini', 'vllm'], 
                       help='LLM type: gemini, openai (for vLLM), or vllm')
    parser.add_argument('--api_base', type=str, default='http://localhost:8040/v1',
                       help='API base URL (for vLLM/OpenAI API)')
    parser.add_argument('--api_key', type=str, default='None',
                       help='API key (for vLLM, can be any value)')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B-Instruct-2507',
                       help='Model name (for vLLM)')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max tokens')
    parser.add_argument('--output_dir', type=str, default='../output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--save_steps', action='store_true', default=True, help='Save step-by-step logs')
    # 已註解：SoM 功能暫時不使用
    # parser.add_argument('--enable_som', action='store_true', default=True, help='Enable Set-of-Marks visualization')
    # parser.add_argument('--som_scheme', type=str, default='region_id', 
    #                    choices=['region_id', 'object_id', 'sequential'],
    #                    help='SoM numbering scheme')
    # parser.add_argument('--som_alpha', type=int, default=120, help='SoM mask transparency (0-255)')
    parser.add_argument('--force_llm_category', action='store_true', default=True, 
                       help='Force LLM-based category detection even if dataset provides categories')
    args = parser.parse_args()
    
    # 設置認證（僅 Gemini 需要）
    if args.llm_type == 'gemini':
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            credential_paths = [
                '../gen-lang-client-0647114394-36f976d459c4.json'
            ]
            for path in credential_paths:
                if os.path.exists(path):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(path)
                    break
    
    # 創建 LLM 客戶端
    if args.llm_type == 'gemini':
        if not args.project_id:
            raise ValueError("--project_id is required for Gemini")
        llm_client = create_llm_client(
            client_type='gemini',
            project_id=args.project_id,
            location=args.location,
            think_mode=args.think_mode
        )
        print("使用 Gemini (Vertex AI)")

    if args.llm_type == 'vllm':
        # 自動檢測是否為視覺模型
        model_name_lower = args.model_name.lower()
        is_vision = any(keyword in model_name_lower for keyword in ['vl', 'vision', 'visual'])
        
        llm_client = create_llm_client(
            client_type='vllm',
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            is_vision_model=is_vision
        )
        print(f"使用 vLLM 本地伺服器: {args.api_base}")
        print(f"模型: {args.model_name}")
        print(f"模型類型: {'視覺語言模型' if is_vision else '純文本模型'}")


    else:
        raise ValueError(f"不支援的 LLM 類型: {args.llm_type}")

    
    # 根據選擇的數據集設置路徑
    dataset = args.dataset
    base_dir = '/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/data/'
    if dataset == 'train':
        data_file = os.path.join(base_dir, 'train.json')
        image_dir = os.path.join(base_dir, 'train', 'images')
    elif dataset == 'test':
        data_file = os.path.join(base_dir, 'test.json')
        image_dir = os.path.join(base_dir, 'test', 'images')
    elif dataset == 'val':
        data_file = os.path.join(base_dir, 'val.json')
        image_dir = os.path.join(base_dir, 'val', 'images')
    else:
        raise ValueError(f"不支援的數據集: {dataset}")
    
    # 載入數據集
    print(f"載入 {dataset} set...")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"數據文件不存在: {data_file}")
    
    with open(data_file, 'r') as f:
        dataset_data = json.load(f)
    
    # 過濾有圖片的項目
    valid_data = []
    for item in dataset_data:
        image_path = os.path.join(image_dir, item['image'])
        if os.path.exists(image_path):
            valid_data.append(item)
        if len(valid_data) >= args.limit:
            break
    
    print(f"找到 {len(valid_data)} 個有圖片的項目（限制 {args.limit} 筆）")
    
    # 初始化工具
    tools = tools_api(
        dist_model_cfg={'model_path': '../distance_est/ckpt/epoch_5_iter_6831.pth'},
        # inside_model_cfg={'model_path': '../inside_pred/ckpt/epoch_4.pth'},
        inside_model_cfg={
        'model_path': '../inside_pred/ckpt_dual_stream/best_model.pth',  # ← 新模型路徑
        'use_geometry': True,      # ← 啟用 geometric features
        'input_channels': 5,       # RGB(3) + obj_mask(1) + buffer_mask(1)
        'num_geo_features': 8      # IoU, area_ratios, center_coords, depth_diff
        },
        small_dist_model_cfg={'model_path': '../distance_est/ckpt/3m_epoch6.pth'},
        resize=(360, 640),
        mask_IoU_thres=0.3, inside_thres=0.5,
        cascade_dist_thres=300, clamp_distance_thres=25
    )
    
    # 已註解：SoM 功能暫時不使用
    # # 初始化 SoM 可視化器（如果啟用）
    # som_visualizer = None
    # if args.enable_som:
    #     som_visualizer = SoMVisualizer(
    #         font_size=25, 
    #         mask_alpha=args.som_alpha
    #     )
    #     # 創建輸出目錄
    #     os.makedirs(os.path.join(args.output_dir, 'som_visualizations'), exist_ok=True)
    #     print(f"SoM 可視化已啟用，使用 {args.som_scheme} 編號方案")
    som_visualizer = None
    
    # 處理每個項目
    results = []
    correct_count = 0
    has_ground_truth = False  # 檢查是否有 ground truth
    category_correct_count = 0  # 分類正確數
    category_total_count = 0  # 有 GT 分類的總數
    
    for item in tqdm(valid_data, desc="Processing"):
        item_id = item['id']
        ground_truth = item.get('normalized_answer', '')
        
        # 檢查是否有 ground truth（用於判斷是否需要計算準確率）
        if ground_truth and not has_ground_truth:
            has_ground_truth = True
        
        # 將 image_dir 添加到 item 中，以便 Agent 使用
        item['image_dir'] = image_dir
        
        # 可視化 debug（如果需要）
        if len(results) == 0:  # 只對第一個樣本可視化
            try:
                import sys
                sys.path.append('../utils')
                from utils.visualize import visualize_masks_and_depth
                
                image_path = os.path.join(image_dir, item['image'])
                if not os.path.exists(image_path):
                    print(f"Visualization skipped, image not found: {image_path}")
                else:
                    depth_rel = item.get('depth') or item.get('depth_image')
                    if not depth_rel:
                        print("Visualization skipped, no depth path provided.")
                    else:
                        depth_path = os.path.join(image_dir, depth_rel)
                        if not os.path.exists(depth_path):
                            print(f"Visualization skipped, depth image not found: {depth_path}")
                        else:
                            output_path = os.path.join(args.output_dir, f"debug_visualization_{item_id}.png")
                            print(f"\nVisualizing masks for item {item_id}")
                            visualize_masks_and_depth(item['rle'], image_path, depth_path, output_path)
                            print(f"Saved visualization to {output_path}")
            except Exception as vis_e:
                print(f"Visualization failed: {vis_e}")
        
        try:
            agent = Agent(llm_client, tools, item, 
                         think_mode=args.think_mode, 
                         verbose=args.verbose,
                         # som_visualizer=som_visualizer,  # 已註解：SoM 功能暫時不使用
                         output_dir=args.output_dir,
                         force_llm_category=args.force_llm_category)
            
            try:
                agent.set_masks()
                raw_answer = agent.set_question()
                formatted_answer = agent.format_answer(raw_answer)
            except Exception as agent_exc:
                # 記錄 agent 執行錯誤
                formatted_answer = '-1'
                error_msg = f"{type(agent_exc).__name__}: {str(agent_exc)}"
                print(f"[ERROR] Item {item_id}: {error_msg}")
            
            # 如果有 ground truth，進行答案比較
            if has_ground_truth and ground_truth:
                is_correct_strict = normalize_answer(formatted_answer, ground_truth, tolerance_percent=0.1)  # 嚴格模式
                is_correct_10 = normalize_answer(formatted_answer, ground_truth, tolerance_percent=10.0)     # 10% 容錯
                is_correct_20 = normalize_answer(formatted_answer, ground_truth, tolerance_percent=20.0)     # 20% 容錯
                
                if is_correct_10:
                    correct_count += 1
            else:
                # 沒有 ground truth，設置為 None
                is_correct_strict = None
                is_correct_10 = None
                is_correct_20 = None
            
            # 檢查分類準確率（如果有 GT category）
            if agent.ground_truth_category:
                category_total_count += 1
                if agent.predicted_category == agent.ground_truth_category:
                    category_correct_count += 1
                else:
                    # 記錄分類錯誤
                    if args.verbose or args.force_llm_category:
                        print(f"[CATEGORY MISMATCH] ID: {item_id}")
                        print(f"  Predicted: {agent.predicted_category}")
                        print(f"  Ground Truth: {agent.ground_truth_category}")
                        print(f"  Question: {item['conversations'][0]['value'][:100]}")
            
            result_entry = {
                'id': item_id,
                'predicted': str(formatted_answer),
                'ground_truth': str(ground_truth) if ground_truth else None,
                'predicted_category': agent.predicted_category,  # LLM predicted category
                'ground_truth_category': agent.ground_truth_category,  # Dataset GT category
                'category': agent.ground_truth_category or agent.predicted_category,  # For stats use GT if available
                'category_correct': agent.predicted_category == agent.ground_truth_category if agent.ground_truth_category else None,
                'conversation': agent.messages if 'agent' in locals() else [],
                'question': item['conversations'][0]['value']
            }
            
            # 如果有 ground truth，添加準確率信息
            if has_ground_truth and ground_truth:
                result_entry['correct'] = is_correct_10
                result_entry['correct_strict'] = is_correct_strict
                result_entry['correct_10'] = is_correct_10
                result_entry['correct_20'] = is_correct_20
            
            # 添加步驟日誌
            if args.save_steps and 'agent' in locals():
                result_entry['step_log'] = agent.step_log
            else:
                result_entry['step_log'] = []
            
            results.append(result_entry)
            
        except Exception as e:
            print(f"[ERROR] Fatal error processing item {item_id}: {str(e)}")
            result_entry = {
                'id': item_id,
                'predicted': '-1',
                'ground_truth': str(ground_truth) if ground_truth else None,
                'category': item.get('category', 'unknown'),
                'error': str(e),
                'error_type': type(e).__name__,
                'conversation': [],
                'question': item['conversations'][0]['value'],
                'step_log': []
            }
            
            if has_ground_truth and ground_truth:
                result_entry['correct'] = False
                result_entry['correct_strict'] = False
                result_entry['correct_10'] = False
                result_entry['correct_20'] = False
            
            results.append(result_entry)
    
    # 如果有 ground truth，計算準確率
    if has_ground_truth:
        strict_count = sum(1 for r in results if r.get('correct_strict', False))
        correct_10_count = sum(1 for r in results if r.get('correct_10', False))
        correct_20_count = sum(1 for r in results if r.get('correct_20', False))
        
        accuracy_strict = strict_count / len(results) * 100
        accuracy_10 = correct_10_count / len(results) * 100
        accuracy_20 = correct_20_count / len(results) * 100
        
        print(f"\n準確率報告 ({dataset} set):")
        print(f"嚴格模式 (±0.1): {accuracy_strict:.2f}% ({strict_count}/{len(results)})")
        print(f"10% 容錯模式: {accuracy_10:.2f}% ({correct_10_count}/{len(results)})")
        print(f"20% 容錯模式: {accuracy_20:.2f}% ({correct_20_count}/{len(results)})")
        print(f"預設使用: 10% 容錯模式")
        
        # 顯示分類準確率
        if category_total_count > 0:
            category_accuracy = category_correct_count / category_total_count * 100
            print(f"\n問題分類準確率: {category_accuracy:.2f}% ({category_correct_count}/{category_total_count})")
            
            # 按類別顯示分類準確率
            if args.force_llm_category or args.verbose:
                print(f"\n分類準確率（按類別）:")
                for true_cat in ['left_right', 'count', 'distance', 'mcq']:
                    cat_items = [r for r in results if r.get('ground_truth_category') == true_cat]
                    if cat_items:
                        correct = sum(1 for r in cat_items if r.get('category_correct', False))
                        acc = correct / len(cat_items) * 100
                        print(f"  {true_cat}: {acc:.2f}% ({correct}/{len(cat_items)})")
        
        # 分析錯誤
        analyze_errors(results, args.output_dir, dataset)
    else:
        print(f"\n{dataset} set 沒有 ground truth，跳過準確率計算")
        print(f"處理了 {len(results)} 個樣本")
    
    # 保存結果
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{dataset}_eval_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"結果已保存到: {output_file}")
    
    # 如果是 test set，額外生成 predictions.json
    if dataset == 'test':
        predictions = []
        for result in results:
            predicted = result.get('predicted', '-1')
            predicted_category = result.get('predicted_category', 'mcq')
            
            # 根據分類確定答案類型
            normalized_answer = None
            predicted_str = str(predicted).strip()
            
            # left_right 和 mcq 應該是字符串
            if predicted_category in ['left_right', 'mcq']:
                normalized_answer = predicted_str
            # count 和 distance 應該是數字
            elif predicted_category in ['count', 'distance']:
                try:
                    float_val = float(predicted_str)
                    if predicted_category == 'count' or float_val.is_integer():
                        normalized_answer = int(float_val)
                    else:
                        normalized_answer = float_val
                except (ValueError, TypeError):
                    # 如果轉換失敗，保持字符串
                    normalized_answer = predicted_str
            else:
                # 未知類別，嘗試自動判斷
                try:
                    float_val = float(predicted_str)
                    if '.' not in predicted_str or float_val.is_integer():
                        normalized_answer = int(float_val)
                    else:
                        normalized_answer = float_val
                except (ValueError, TypeError):
                    normalized_answer = predicted_str
            
            predictions.append({
                'id': result.get('id', ''),
                'normalized_answer': normalized_answer
            })
        
        prediction_file = os.path.join(args.output_dir, 'predictions.json')
        with open(prediction_file, 'w') as f:
            json.dump(predictions, f, indent=4)
        print(f"預測結果已保存到: {prediction_file}")
        
        # 額外輸出一個帶分類資訊的版本（用於除錯）
        debug_file = os.path.join(args.output_dir, 'predictions_with_category.json')
        with open(debug_file, 'w') as f:
            debug_predictions = [
                {
                    'id': r.get('id', ''),
                    'normalized_answer': p['normalized_answer'],
                    'predicted_category': r.get('predicted_category'),
                    'answer_type': type(p['normalized_answer']).__name__
                }
                for r, p in zip(results, predictions)
            ]
            json.dump(debug_predictions, f, indent=4)
        print(f"除錯用預測結果（含分類）已保存到: {debug_file}")

if __name__ == "__main__":
    main()

