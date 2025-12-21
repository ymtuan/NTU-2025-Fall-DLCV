#!/usr/bin/env python3
"""
自动化错误分析工具
使用 VLM 分析预测错误，识别出错的工具函数
"""

import json
import os
import argparse
import sys
from typing import Dict, List, Optional
import re

# 添加 agent 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agent'))
from llm_client import create_llm_client


def load_json_file(file_path: str) -> any:
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_image_in_test_json(test_data: List[Dict], item_id: str) -> Optional[str]:
    """在 test.json 中查找对应的 image 字段"""
    for item in test_data:
        if item.get('id') == item_id:
            return item.get('image')
    return None


def build_annotated_image_path(image_filename: str, base_dir: str = None) -> str:
    """构建 annotated image 路径"""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), 'data', 'test', 'images_annotated')
    
    # 移除 .png 扩展名（如果有），然后添加 _annotated.png
    image_name = image_filename.replace('.png', '')
    annotated_filename = f"{image_name}_annotated.png"
    return os.path.join(base_dir, annotated_filename)


def extract_conversation_from_second(prediction_item: Dict) -> List[Dict]:
    """提取 conversation，从第二个消息开始"""
    conversation = prediction_item.get('conversation', [])
    if len(conversation) > 1:
        return conversation[1:]  # 跳过第一个很长的 user prompt
    return conversation


def format_conversation_for_prompt(conversation: List[Dict]) -> str:
    """格式化 conversation 为可读的文本"""
    formatted = []
    for msg in conversation:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        # 截断过长的内容
        if len(content) > 500:
            content = content[:500] + "..."
        formatted.append(f"{role.upper()}: {content}")
    return "\n".join(formatted)


def build_vlm_prompt(prediction_item: Dict, conversation_text: str) -> str:
    """构建 VLM 分析 prompt"""
    question = prediction_item.get('question', '')
    predicted_answer = prediction_item.get('predicted_answer', '')
    ground_truth = prediction_item.get('ground_truth_from_eval', '')
    freeform_answer = prediction_item.get('freeform_answer', '')
    
    prompt = f"""你是一个错误分析专家。请分析以下空间推理任务的错误案例。

问题: {question}

预测答案: {predicted_answer}
正确答案: {ground_truth}
"""
    
    if freeform_answer:
        prompt += f"\n正确答案的详细解释: {freeform_answer}\n"
    
    prompt += f"""
对话记录（从第二个消息开始）:
{conversation_text}

请仔细分析对话记录中的工具调用和结果，判断是哪个工具函数出错导致预测错误。

工具函数包括：
- distance: dist() - 计算两个 mask 之间的距离
- is_closest: closest() - 找到最近的 mask
- inclusion: inside() - 计算包含关系
- other: 其他函数（is_left, is_right, most_left, most_right, is_empty, middle）

重要：请确保 error_tool 字段的值必须与你在 explanation 中分析的错误工具一致。

请严格按照以下 JSON 格式输出，不要添加任何其他文字或解释：
{{
  "error_tool": "distance",
  "explanation": "详细的错误解释"
}}

或者

{{
  "error_tool": "is_closest",
  "explanation": "详细的错误解释"
}}

或者

{{
  "error_tool": "inclusion",
  "explanation": "详细的错误解释"
}}

或者

{{
  "error_tool": "other",
  "explanation": "详细的错误解释"
}}

注意：
1. error_tool 必须是以下四个值之一：distance, is_closest, inclusion, other
2. explanation 中如果提到某个工具出错，error_tool 必须对应那个工具
3. 只输出 JSON，不要有其他文字
"""
    return prompt


def parse_vlm_response(response: str) -> Dict[str, str]:
    """解析 VLM 返回的 JSON 响应，并验证 error_tool 与 explanation 的一致性"""
    # 清理响应文本，移除可能的 markdown 代码块标记
    cleaned_response = response.strip()
    if cleaned_response.startswith('```'):
        # 移除 markdown 代码块标记
        lines = cleaned_response.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        cleaned_response = '\n'.join(lines)
    
    # 尝试提取 JSON（支持多行）
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_matches = re.findall(json_pattern, cleaned_response, re.DOTALL)
    
    valid_tools = ['distance', 'is_closest', 'inclusion', 'other']
    tool_keywords = {
        'distance': ['dist(', 'distance', '距离'],
        'is_closest': ['closest(', '最近', 'nearest'],
        'inclusion': ['inside(', '包含', 'inclusion', 'within'],
        'other': ['is_left(', 'is_right(', 'most_left(', 'most_right(', 'is_empty(', 'middle(']
    }
    
    for json_str in json_matches:
        try:
            result = json.loads(json_str)
            error_tool = result.get('error_tool', '').strip().lower()
            explanation_raw = result.get('explanation', '').strip()
            explanation_lower = explanation_raw.lower()
            
            # 验证 error_tool 是否有效
            if error_tool in valid_tools:
                # 验证一致性：检查 explanation 中是否明确提到其他工具出错
                # 优先检查 explanation 中明确提到的错误工具
                detected_tool = None
                max_confidence = 0
                
                for tool, keywords in tool_keywords.items():
                    confidence = 0
                    # 检查是否有明确的错误描述配合工具名称
                    error_indicators = ['错误', 'error', 'wrong', 'incorrect', 'failed', '失败', '问题', '根源', 'root cause']
                    
                    for kw in keywords:
                        if kw in explanation_lower:
                            confidence += 1
                            # 检查附近是否有错误指示词
                            kw_pos = explanation_lower.find(kw)
                            context = explanation_lower[max(0, kw_pos-50):kw_pos+100]
                            if any(indicator in context for indicator in error_indicators):
                                confidence += 2  # 高置信度
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
                        detected_tool = tool
                
                # 如果检测到的工具与 error_tool 不一致，且置信度足够高，则修正
                if detected_tool and detected_tool != error_tool and max_confidence >= 2:
                    print(f"Warning: error_tool={error_tool} but explanation clearly mentions {detected_tool} error (confidence={max_confidence}). Correcting to {detected_tool}")
                    error_tool = detected_tool
                
                return {
                    'error_tool': error_tool,
                    'explanation': explanation_raw  # 使用原始 explanation
                }
        except json.JSONDecodeError:
            continue
    
    # 如果没有找到有效的 JSON，尝试直接解析整个响应
    try:
        result = json.loads(cleaned_response)
        error_tool = result.get('error_tool', '').strip().lower()
        explanation = result.get('explanation', '').strip()
        
        valid_tools = ['distance', 'is_closest', 'inclusion', 'other']
        if error_tool in valid_tools:
            return {
                'error_tool': error_tool,
                'explanation': explanation
            }
    except json.JSONDecodeError:
        pass
    
    # 如果都失败了，返回 unknown
    return {
        'error_tool': 'unknown',
        'explanation': f'Could not parse JSON from response. Response preview: {response[:300]}'
    }


def analyze_single_case(
    prediction_item: Dict,
    test_data: List[Dict],
    llm_client,
    base_dir: str = None
) -> Optional[Dict]:
    """分析单个错误案例"""
    item_id = prediction_item.get('id')
    if not item_id:
        print(f"Warning: Item missing id, skipping")
        return None
    
    # 查找对应的 image
    image_filename = find_image_in_test_json(test_data, item_id)
    if not image_filename:
        print(f"Warning: Could not find image for id {item_id}, skipping")
        return None
    
    # 构建 annotated image 路径
    annotated_image_path = build_annotated_image_path(image_filename, base_dir)
    
    # 检查文件是否存在
    if not os.path.exists(annotated_image_path):
        print(f"Warning: Annotated image not found: {annotated_image_path}, skipping")
        return None
    
    # 提取 conversation（从第二个开始）
    conversation = extract_conversation_from_second(prediction_item)
    conversation_text = format_conversation_for_prompt(conversation)
    
    # 构建 prompt
    prompt = build_vlm_prompt(prediction_item, conversation_text)
    
    try:
        # 更新图片
        llm_client.update_image(annotated_image_path)
        
        # 重置对话（每次分析都是独立的）
        llm_client.reset_conversation()
        
        # 调用 VLM
        response = llm_client.send_message(prompt, is_classification=False)
        
        # 解析响应
        analysis_result = parse_vlm_response(response)
        
        # 构建输出结果
        result = {
            'id': item_id,
            'error_tool': analysis_result['error_tool'],
            'explanation': analysis_result['explanation'],
            'annotated_image_path': annotated_image_path,
            'predicted_answer': prediction_item.get('predicted_answer', ''),
            'ground_truth': prediction_item.get('ground_truth_from_eval', ''),
            'question': prediction_item.get('question', ''),
            'conversation': conversation,
            'category': prediction_item.get('category_from_eval', '')
        }
        
        return result
        
    except Exception as e:
        print(f"Error analyzing case {item_id}: {str(e)}")
        return {
            'id': item_id,
            'error_tool': 'unknown',
            'explanation': f'VLM call failed: {str(e)}',
            'annotated_image_path': annotated_image_path,
            'predicted_answer': prediction_item.get('predicted_answer', ''),
            'ground_truth': prediction_item.get('ground_truth_from_eval', ''),
            'question': prediction_item.get('question', ''),
            'conversation': conversation,
            'category': prediction_item.get('category_from_eval', '')
        }


def main():
    parser = argparse.ArgumentParser(description='使用 VLM 分析预测错误')
    parser.add_argument(
        '--predictions_file',
        type=str,
        default='/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/output/log_mse_2/predictions_detail_filtered.json',
        help='预测结果文件路径'
    )
    parser.add_argument(
        '--test_json',
        type=str,
        default='/tmp1/d13944024_home/kai/dlcv_final/SpatialAgent/data/test.json',
        help='test.json 文件路径'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='output/log_mse_2/error_analysis.json',
        help='输出文件路径'
    )
    parser.add_argument(
        '--api_base',
        type=str,
        required=True,
        help='VLM API base URL'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='VLM 模型名称'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default='dummy',
        help='API key（可选）'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理的案例数量（用于测试）'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default=None,
        help='项目基础目录（默认使用脚本所在目录）'
    )
    
    args = parser.parse_args()
    
    # 设置基础目录
    if args.base_dir is None:
        args.base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建完整路径
    predictions_file = os.path.join(args.base_dir, args.predictions_file)
    test_json_file = os.path.join(args.base_dir, args.test_json)
    output_file = os.path.join(args.base_dir, args.output_file)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Loading predictions from: {predictions_file}")
    predictions_data = load_json_file(predictions_file)
    print(f"Found {len(predictions_data)} error cases")
    
    print(f"Loading test.json from: {test_json_file}")
    test_data = load_json_file(test_json_file)
    print(f"Loaded {len(test_data)} test cases")
    
    # 创建 LLM client
    print(f"Creating VLM client: {args.model_name}")
    llm_client = create_llm_client(
        client_type='vllm',
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model_name,
        temperature=0.2,
        max_tokens=1024,
        is_vision_model=True
    )
    
    # 处理每个案例
    results = []
    limit = args.limit if args.limit else len(predictions_data)
    
    for i, prediction_item in enumerate(predictions_data[:limit]):
        print(f"\nProcessing case {i+1}/{limit}: {prediction_item.get('id', 'unknown')}")
        
        result = analyze_single_case(
            prediction_item,
            test_data,
            llm_client,
            base_dir=os.path.join(args.base_dir, 'data', 'test', 'images_annotated')
        )
        
        if result:
            results.append(result)
            print(f"  -> Error tool: {result['error_tool']}")
        else:
            print(f"  -> Skipped")
    
    # 保存结果
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompleted! Analyzed {len(results)} cases")
    print(f"Results saved to: {output_file}")
    
    # 统计错误工具分布
    tool_counts = {}
    for result in results:
        tool = result.get('error_tool', 'unknown')
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    print("\nError tool distribution:")
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count}")


if __name__ == '__main__':
    main()

