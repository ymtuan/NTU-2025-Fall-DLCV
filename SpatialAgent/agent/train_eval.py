#!/usr/bin/env python3
"""
在 train set 上評估並分析錯誤
"""

import re
import json
import os
import ast
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict, Counter
from tools import tools_api
from mask import parse_masks_from_conversation
from llm_client import create_llm_client
from som_visualizer import SoMVisualizer

class Agent:
    def __init__(self, llm_client, tools_api, input_item, think_mode=False, verbose=False, som_visualizer=None, output_dir='../output'):
        self.llm_client = llm_client
        self.tools_api = tools_api
        self.verbose = verbose
        self.step_log = []  # 記錄每個步驟
        self.som_visualizer = som_visualizer
        self.output_dir = output_dir
        
        self.messages = []
        self.prompt_preamble = open('prompt/agent_example.txt', 'r').read()
        self.answer_preamble = open('prompt/answer.txt', 'r').read()
        self.input = input_item
        self.masks = None
        self.question = None
        self.som_image_path = None  # SoM 處理後的圖像路徑
        self.original_image_path = None  # 原始圖像路徑
    
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
            # 默認使用 train 路徑（向後兼容）
            self.original_image_path = '/tmp1/d13944024_home/kai/dlcv_final/DLCV_Final1/train/images/' + self.input['image']
        
        # 如果有 SoM 可視化器，生成 SoM 圖像
        if self.som_visualizer:
            self._generate_som_image()
    
    def _generate_som_image(self):
        """生成 SoM 可視化圖像"""
        try:
            import os
            som_dir = os.path.join(self.output_dir, 'som_images')
            os.makedirs(som_dir, exist_ok=True)
            
            self.som_image_path = os.path.join(som_dir, f"som_{self.input['id']}.png")
            
            self.som_visualizer.create_som_visualization(
                image_path=self.original_image_path,
                masks=self.masks,
                output_path=self.som_image_path,
                numbering_scheme="region_id"
            )
            
            self._log_step('som_image_generated', {
                'som_image_path': self.som_image_path
            })
            
        except Exception as e:
            if self.verbose:
                print(f"SoM 圖像生成失敗: {e}")
            self.som_image_path = self.original_image_path
    
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
    
    def _is_direction_question(self, command, result):
        """檢查是否是方向問題"""
        import re
        func_match = re.match(r"(\w+)\s*\(", command.strip())
        if not func_match:
            return False
        
        func_name = func_match.group(1)
        return func_name in ['is_left', 'is_right'] and isinstance(result, bool)
    
    def _convert_boolean_to_direction(self, command, boolean_result):
        """將 Boolean 結果轉換為方向字符串"""
        import re
        func_match = re.match(r"(\w+)\s*\(", command.strip())
        if not func_match:
            return str(boolean_result)
        
        func_name = func_match.group(1)
        
        if func_name == 'is_left':
            return 'left' if boolean_result else 'right'
        elif func_name == 'is_right':
            return 'right' if boolean_result else 'left'
        
        return str(boolean_result)
    
    def format_answer(self, raw_answer=None):
        """格式化答案，可以接受來自 conversation_loop 的答案或重新查詢 LLM"""
        
        if raw_answer is not None:
            # 使用提供的答案（來自 conversation_loop）
            answer = raw_answer.strip()
            
            self._log_step('format_answer_from_conversation', {
                'raw_answer': answer
            })
        else:
            # 重新查詢 LLM（舊行為）
            self._log_step('format_answer_start', {
                'answer_preamble': self.answer_preamble[:100]
            })
            
            answer = self.llm_client.send_message(self.answer_preamble)
            answer = answer.strip()
            
            self._log_step('format_answer_response', {
                'raw_answer': answer
            })
        
        # 嘗試從答案中提取數值或關鍵詞
        formatted_answer = self._extract_final_answer(answer)
        
        # 如果答案是 mask 名稱，需要根據問題類型決定如何處理
        if formatted_answer in self.masks:
            # 檢查原問題來決定返回格式
            original_question = self.input['conversations'][0]['value'].lower()
            
            if any(word in original_question for word in ['which', 'what', 'who']):
                # 對於 "which" 類型問題，返回 object_id
                result = self.masks[formatted_answer].object_id
                self._log_step('format_answer_mapped_to_object_id', {
                    'original': formatted_answer,
                    'mapped_to': result,
                    'reason': 'Which-type question, using object_id'
                })
            else:
                # 對於其他問題，保持原 mask 名稱
                result = formatted_answer
                self._log_step('format_answer_kept_mask_name', {
                    'answer': result,
                    'reason': 'Non-which question, keeping mask name'
                })
            
            return result
        else:
            self._log_step('format_answer_direct', {
                'answer': formatted_answer
            })
            return formatted_answer
    
    def _extract_final_answer(self, answer_text):
        """從答案文本中提取最終答案"""
        import re
        
        # 先嘗試尋找距離相關的數字模式
        distance_patterns = [
            r'(?:distance|dist).*?is\s+(\d+\.?\d*)',  # "distance is 4.9"
            r'(\d+\.?\d*)\s*(?:units?|meters?|m\b)',   # "4.9 units" or "4.9 meters"
            r'(?:is|=)\s*(\d+\.?\d*)',                 # "is 4.9" or "= 4.9"
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
        mask_match = re.search(r'\b(buffer_\d+|pallet_\d+|transporter_\d+|object_\d+)\b', answer_text)
        if mask_match:
            return mask_match.group(1)
        
        # 最後才尋找任意數字（避免誤匹配 object IDs）
        # 但排除明顯是物件ID的數字
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', answer_text)
        if numbers:
            # 過濾掉可能是物件ID的數字（通常是小整數且前面有下劃線）
            filtered_numbers = []
            for num in numbers:
                # 檢查這個數字前面是否有下劃線（表示是物件ID）
                num_pos = answer_text.find(num)
                if num_pos > 0 and answer_text[num_pos-1] == '_':
                    continue  # 跳過物件ID
                filtered_numbers.append(num)
            
            if filtered_numbers:
                return filtered_numbers[-1]  # 返回最後一個非ID數字
        
        # 如果都找不到，返回原答案
        return answer_text

    def set_question(self):
        self.messages = []
        original_question = self.input['conversations'][0]['value']
        
        # 預處理問題，將 <mask> 替換為具體代碼
        self.question = self._preprocess_question(original_question)
        
        self._log_step('set_question_start', {
            'original_question': original_question,
            'processed_question': self.question,
            'image': self.input['image']
        })
        
        full_prompt = self.prompt_preamble.replace("<question>", self.question)
        
        # 加入可用 masks 資訊到 prompt
        available_masks = ", ".join(self.masks.keys())
        full_prompt += f"\n\nAvailable masks in this image: {available_masks}"
        
        # 使用 SoM 圖像（如果有）或原始圖像給 VLM
        image_for_vlm = self.som_image_path if self.som_image_path else self.original_image_path
        
        self._log_step('prompt_prepared', {
            'prompt_length': len(full_prompt),
            'available_masks': available_masks,
            'vlm_image_path': image_for_vlm,
            'original_image_path': self.original_image_path
        })
        
        self.messages.append({"role": "user", "content": full_prompt})
        
        # 重要：tools_api 仍使用原始圖像進行距離等計算
        self.tools_api.update_image(self.original_image_path)
        
        # VLM 客戶端使用 SoM 圖像
        if hasattr(self.llm_client, 'update_image'):
            self.llm_client.update_image(image_for_vlm)
        
        return self._conversation_loop()

    def _conversation_loop(self, budget=10):
        usage = 0
        execute_flag = False
        last_tool_result = None  # 保存上一次工具執行結果
        last_tool_command = None  # 保存上一次工具命令
        # 重置 LLM 客戶端的對話歷史
        self.llm_client.reset_conversation()
        
        self._log_step('conversation_loop_start', {
            'budget': budget,
            'initial_message_count': len(self.messages)
        })
        
        while usage < budget:
            usage += 1
            
            self._log_step(f'iteration_{usage}_start', {
                'iteration': usage,
                'message_count': len(self.messages),
                'execute_flag': execute_flag
            })
            
            latest_message = self.messages[-1]["content"]
            
            self._log_step(f'iteration_{usage}_send_to_llm', {
                'message_length': len(latest_message),
                'message_preview': latest_message
            })
            
            response_text = self.llm_client.send_message(latest_message)
            
            self._log_step(f'iteration_{usage}_llm_response', {
                'response_length': len(response_text),
                'response_preview': response_text
            })
            
            # llm_client.send_message 已經更新了它的 messages，我們也更新自己的
            self.messages = self.llm_client.messages.copy()

            # 檢查是否有 <final> tag（LLM 標記工具結果為最終答案）
            final_match = re.search(r"<final>", response_text, re.IGNORECASE)
            if final_match and last_tool_result is not None:
                # LLM 判斷工具結果是最終答案，處理並返回
                # 特殊處理方向函數的 Boolean 結果
                if self._is_direction_question(last_tool_command, last_tool_result):
                    direction_result = self._convert_boolean_to_direction(last_tool_command, last_tool_result)
                    self._log_step('direction_result_converted', {
                        'command': last_tool_command,
                        'boolean_result': str(last_tool_result),
                        'direction_result': direction_result,
                        'reason': 'LLM marked as final, converted boolean to direction string'
                    })
                    return direction_result
                
                self._log_step('final_tool_result', {
                    'command': last_tool_command,
                    'result': str(last_tool_result),
                    'reason': 'LLM marked tool result as final answer'
                })
                return str(last_tool_result)

            execute_match = re.search(r"<execute>(.*?)</execute>", response_text, re.DOTALL)
            
            # 如果沒有找到完整的 execute 標籤，嘗試找不完整的
            if not execute_match:
                incomplete_execute = re.search(r"<execute>\s*$", response_text, re.DOTALL)
                if incomplete_execute:
                    # LLM 開始了 execute 但沒有完成，提示它繼續
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
                
                result = self._execute_function(command)
                
                # 保存工具結果和命令，以便 LLM 標記 <final> 時使用
                last_tool_result = result
                last_tool_command = command
                
                self._log_step(f'iteration_{usage}_execute_result', {
                    'result': str(result),
                    'result_type': type(result).__name__
                })
                
                # 將工具結果返回給 LLM，讓 LLM 判斷是否是最終答案
                # LLM 可以輸出 <final> tag 來標記這是最終答案
                self.messages.append({"role": "user", "content": f"Tool result: {result}"})
                continue

            answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
            # 如果沒有找到完整的答案標籤，嘗試找只有開始標籤的情況
            if not answer_match:
                answer_start_match = re.search(r"<answer>\s*(.*?)$", response_text, re.DOTALL | re.MULTILINE)
                if answer_start_match:
                    answer_match = answer_start_match
                    
            if answer_match and execute_flag:
                final_answer = answer_match.group(1).strip()
                
                self._log_step('final_answer_found', {
                    'answer': final_answer,
                    'total_iterations': usage
                })
                
                self.messages.append({"role": "assistant", "content": final_answer})
                return final_answer

            self._log_step(f'iteration_{usage}_no_valid_action', {
                'response_preview': response_text,
                'has_execute': execute_match is not None,
                'has_answer': answer_match is not None
            })
            
            print("No valid action found. Ending interaction.")
            raise ValueError("No valid action found in the assistant's response.")

    def _execute_function(self, command):
        original_command = command
        
        # 處理多行命令
        lines = [line.strip() for line in command.split('\n') if line.strip()]
        
        self._log_step('parse_function_call', {
            'original_command': original_command,
            'command_lines': lines
        })
        
        # 如果是多行命令，需要順序執行並保存中間結果
        if len(lines) > 1:
            return self._execute_multiline_command(lines)
        else:
            return self._execute_single_command(lines[0] if lines else command)
    
    def _execute_multiline_command(self, lines):
        """執行多行命令並保存中間結果"""
        variables = {}  # 保存中間變量
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 檢查是否是賦值語句
            assignment_match = re.match(r'(\w+)\s*=\s*(.+)', line)
            if assignment_match:
                var_name = assignment_match.group(1)
                func_call = assignment_match.group(2).strip()
                
                # 執行函數調用
                result = self._execute_single_command(func_call, variables)
                variables[var_name] = result
                
                self._log_step(f'multiline_step_{i}', {
                    'variable': var_name,
                    'function_call': func_call,
                    'result': str(result)
                })
            else:
                # 直接執行函數調用（最後一行）
                result = self._execute_single_command(line, variables)
                self._log_step('multiline_final_result', {
                    'command': line,
                    'result': str(result)
                })
                return result
        
        # 如果沒有最終的非賦值命令，返回最後一個變量的值
        if variables:
            last_var = list(variables.keys())[-1]
            return variables[last_var]
        
        return None
    
    def _execute_single_command(self, command, variables=None):
        """執行單個命令"""
        if variables is None:
            variables = {}
        
        # 清理命令
        command = command.strip().replace('\\n', '').replace('\n', '').replace('\r', '').replace('<', '').replace('>', '')
        
        self._log_step('execute_single_command', {
            'command': command,
            'available_variables': list(variables.keys())
        })
        
        match = re.match(r"(\w+)\s*\((.*)\)", command)
        if not match:
            raise ValueError(f"Invalid function call: {command}")

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
            raise ValueError(f"Unknown function: {func_name}")

        parsed_args = self._parse_arguments(args_str, variables)
        
        self._log_step('arguments_parsed', {
            'parsed_args': [str(arg) for arg in parsed_args],
            'arg_types': [type(arg).__name__ for arg in parsed_args]
        })
        
        result = func_map[func_name](*parsed_args)
        
        self._log_step('function_executed', {
            'function': func_name,
            'result': str(result),
            'result_type': type(result).__name__
        })
        
        return result

    def _parse_arguments(self, args_str, variables=None):
        if variables is None:
            variables = {}
            
        def resolve(node):
            if isinstance(node, ast.Name):
                key = node.id
                
                # 首先檢查是否是變量
                if key in variables:
                    return variables[key]
                
                # 然後檢查是否是 mask
                if key.startswith('<') and key.endswith('>'):
                    key = key[1:-1]
                if key not in self.masks:
                    print(f"ERROR: Mask '{key}' not found in the provided masks.")
                    print(f"Available mask keys: {list(self.masks.keys())}")
                    print(f"Available variables: {list(variables.keys())}")
                    raise ValueError(f"Mask '{key}' not found in the provided masks.")
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
    
    # 保存錯誤案例
    error_file = os.path.join(output_dir, f'{dataset}_errors.json')
    with open(error_file, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"\n錯誤案例已保存到: {error_file}")
    
    return errors

def main():
    parser = ArgumentParser(description="Evaluate on train/test/val set")
    parser.add_argument('--dataset', type=str, default='train', 
                       choices=['train', 'test', 'val'],
                       help='Dataset to evaluate: train, test, or val')
    parser.add_argument('--project_id', type=str, help='Google Cloud Project ID (for Gemini)')
    parser.add_argument('--location', type=str, default='global', help='Location for Vertex AI')
    parser.add_argument('--think_mode', action='store_true', help='Enable think mode')
    parser.add_argument('--limit', type=int, default=1000, help='Number of samples to process')
    parser.add_argument('--llm_type', type=str, default='vllm', 
                       choices=['gemini', 'openai', 'vllm'], 
                       help='LLM type: gemini, openai (for vLLM), or vllm')
    parser.add_argument('--api_base', type=str, default='http://localhost:8060/v1',
                       help='API base URL (for vLLM/OpenAI API)')
    parser.add_argument('--api_key', type=str, default='dummy',
                       help='API key (for vLLM, can be any value)')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct',
                       help='Model name (for vLLM)')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max tokens')
    parser.add_argument('--output_dir', type=str, default='../output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--save_steps', action='store_true', default=True, help='Save step-by-step logs')
    parser.add_argument('--enable_som', action='store_true', default=True, help='Enable Set-of-Marks visualization')
    parser.add_argument('--som_scheme', type=str, default='region_id', 
                       choices=['region_id', 'object_id', 'sequential'],
                       help='SoM numbering scheme')
    parser.add_argument('--som_alpha', type=int, default=120, help='SoM mask transparency (0-255)')
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
    elif args.llm_type in ['openai', 'vllm']:
        llm_client = create_llm_client(
            client_type='openai',
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        print(f"使用 vLLM/OpenAI API: {args.api_base}")
        print(f"模型: {args.model_name}")
    else:
        raise ValueError(f"不支援的 LLM 類型: {args.llm_type}")
    
    # 根據選擇的數據集設置路徑
    dataset = args.dataset
    base_dir = '/tmp1/d13944024_home/kai/dlcv_final/DLCV_Final1'
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
        inside_model_cfg={'model_path': '../inside_pred/ckpt/epoch_4.pth'},
        small_dist_model_cfg={'model_path': '../distance_est/ckpt/3m_epoch6.pth'},
        resize=(360, 640),
        mask_IoU_thres=0.3, inside_thres=0.5,
        cascade_dist_thres=300, clamp_distance_thres=25
    )
    
    # 初始化 SoM 可視化器（如果啟用）
    som_visualizer = None
    if args.enable_som:
        som_visualizer = SoMVisualizer(
            font_size=25, 
            mask_alpha=args.som_alpha
        )
        # 創建輸出目錄
        os.makedirs(os.path.join(args.output_dir, 'som_visualizations'), exist_ok=True)
        print(f"SoM 可視化已啟用，使用 {args.som_scheme} 編號方案")
    
    # 處理每個項目
    results = []
    correct_count = 0
    has_ground_truth = False  # 檢查是否有 ground truth
    
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
                output_path = os.path.join(args.output_dir, f"debug_visualization_{item_id}.png")
                
                # 簡單可視化 masks
                print(f"\nVisualizing masks for item {item_id}")
                visualize_masks_and_depth(item['rle'], image_path, None, output_path)
                print(f"Saved visualization to {output_path}")
            except Exception as vis_e:
                print(f"Visualization failed: {vis_e}")
        
        try:
            agent = Agent(llm_client, tools, item, 
                         think_mode=args.think_mode, 
                         verbose=args.verbose,
                         som_visualizer=som_visualizer,
                         output_dir=args.output_dir)
            agent.set_masks()
            
            raw_answer = agent.set_question()  # 來自 conversation_loop
            formatted_answer = agent.format_answer(raw_answer)  # 格式化答案
            
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
            
            result_entry = {
                'id': item_id,
                'predicted': str(formatted_answer),
                'ground_truth': str(ground_truth) if ground_truth else None,
                'category': item.get('category', 'unknown'),
                'conversation': agent.messages,
                'question': item['conversations'][0]['value']
            }
            
            # 如果有 ground truth，添加準確率信息
            if has_ground_truth and ground_truth:
                result_entry['correct'] = is_correct_10
                result_entry['correct_strict'] = is_correct_strict
                result_entry['correct_10'] = is_correct_10
                result_entry['correct_20'] = is_correct_20
            
            # 添加步驟日誌
            if args.save_steps:
                result_entry['step_log'] = agent.step_log
            
            results.append(result_entry)
            
        except Exception as e:
            result_entry = {
                'id': item_id,
                'predicted': '-1',
                'ground_truth': str(ground_truth) if ground_truth else None,
                'category': item.get('category', 'unknown'),
                'error': str(e),
                'conversation': [],
                'question': item['conversations'][0]['value']
            }
            
            # 如果有 ground truth，添加準確率信息
            if has_ground_truth and ground_truth:
                result_entry['correct'] = False
                result_entry['correct_strict'] = False
                result_entry['correct_10'] = False
                result_entry['correct_20'] = False
            
            # 添加步驟日誌（如果有）
            if args.save_steps and 'agent' in locals():
                result_entry['step_log'] = agent.step_log
            
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
    
    # 如果是 test set，額外生成 prediction.json
    if dataset == 'test':
        predictions = []
        for result in results:
            predicted = result.get('predicted', '-1')
            
            # 嘗試將 predicted 轉換為適當的類型
            normalized_answer = None
            predicted_str = str(predicted).strip()
            
            # 嘗試轉換為數字（如果是數字字符串）
            try:
                # 先嘗試浮點數（可以處理整數和浮點數）
                float_val = float(predicted_str)
                # 如果是整數（沒有小數點或小數部分為0），轉換為整數
                if '.' not in predicted_str or float_val.is_integer():
                    normalized_answer = int(float_val)
                else:
                    normalized_answer = float_val
            except (ValueError, TypeError):
                # 如果轉換失敗，保持字符串格式（如 "left", "right" 等）
                normalized_answer = predicted_str
            
            predictions.append({
                'id': result.get('id', ''),
                'normalized_answer': normalized_answer
            })
        
        prediction_file = os.path.join(args.output_dir, 'prediction.json')
        with open(prediction_file, 'w') as f:
            json.dump(predictions, f, indent=4)
        print(f"預測結果已保存到: {prediction_file}")

if __name__ == "__main__":
    main()

