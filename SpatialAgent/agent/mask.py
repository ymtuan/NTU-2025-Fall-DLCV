import re
import json
from tqdm import tqdm
from typing import List, Dict
import numpy as np
import pycocotools.mask as mask_utils

class Mask:
    def __init__(self, object_class: str, object_id: int, region_id: int, rle: Dict):
        self.object_class = object_class
        self.object_id = object_id
        self.region_id = region_id
        self.rle = rle
        self.loaded = None
        self.rle['counts'] = self.rle['counts'].encode('utf-8')
    
    def mask_name(self) -> str:
        return f"{self.object_class}_{self.object_id}"

    def region_name(self) -> str:
        return f"region_{self.region_id}"
    
    def decode_mask(self):
        return mask_utils.decode(self.rle).astype(np.float32)

    def __repr__(self):
        return (f"Mask(object_class='{self.object_class}', "
                f"object_id={self.object_id}, "
                f"region_id={self.region_id}) ")

def parse_masks_from_conversation(conversation: str, rle_data: List[Dict]) -> Dict[str, Mask]:
    """
    Parses mask references from the conversation and builds Mask objects,
    storing them in a dictionary with keys like 'pallet_mask_0'.
    
    Supports two formats:
    1. <object_class>_<id> format (e.g., <pallet_0>, <buffer_1>) - used in test set
    2. <mask> format - used in train set, masks are assigned sequentially
    """
    # Regex to find masks: matches <pallet_mask_0>, <transporter_mask_1> etc.
    mask_pattern = re.compile(r"<([a-zA-Z]+)_(\d+)>")
    
    # Regex to find generic <mask> tags
    generic_mask_pattern = re.compile(r"<mask>")

    # Find all matches in the conversation
    matches = mask_pattern.findall(conversation)
    generic_masks = generic_mask_pattern.findall(conversation)

    # Dictionary to store masks
    mask_store: Dict[str, Mask] = {}

    # If we found specific mask references (format 1)
    if matches:
        # Track object ID counters per object class
        object_counters: Dict[str, int] = {}

        for region_id, match in enumerate(matches):
            object_class = match[0]

            # Assign object ID (increment per class)
            object_id = object_counters.get(object_class, 0)
            object_counters[object_class] = object_id + 1

            # Create Mask instance
            mask_obj = Mask(object_class, object_id, region_id, rle_data[region_id])

            # Build key like 'pallet_mask_0'
            mask_key = f"{object_class}_{object_id}"

            # Store in dictionary
            mask_store[mask_key] = mask_obj
    
    # If we only found generic <mask> tags (format 2 - train set)
    elif generic_masks:
        # For train set, parse masks based on the order they appear in the question
        # Pattern: "buffer regions <mask> <mask> <mask> and pallets <mask> <mask>..."
        conversation_lower = conversation.lower()
        
        # Find all <mask> positions and their context
        mask_positions = []
        current_expected_type = None  # Track the current context
        
        for match in generic_mask_pattern.finditer(conversation):
            pos = match.start()
            
            # Look backwards for object type keywords
            context_start = max(0, pos - 100)  # 增加上下文範圍
            context = conversation_lower[context_start:pos]
            
            # Determine object type from immediate context
            obj_type = None
            
            # 檢查最近的類型關鍵字
            type_keywords = ['buffer', 'pallet', 'transporter', 'shelf']
            best_match = None
            best_distance = float('inf')
            
            for keyword in type_keywords:
                # 查找關鍵字 + "masks?" 模式
                patterns = [
                    f'{keyword} masks?',
                    f'{keyword}s?',
                    keyword
                ]
                
                for pattern in patterns:
                    matches = list(re.finditer(pattern, context))
                    if matches:
                        # 找最近的匹配
                        last_match = matches[-1]
                        distance = pos - (context_start + last_match.end())
                        if distance < best_distance:
                            best_distance = distance
                            best_match = keyword
            
            if best_match:
                obj_type = best_match
                current_expected_type = obj_type  # 設置當前期望類型
            elif current_expected_type:
                # 如果沒有明確指示，但有前面的上下文，繼續使用
                obj_type = current_expected_type
            
            mask_positions.append((pos, obj_type))
        
        # Group consecutive masks by object type
        current_type = None
        type_counters = {}
        mask_index = 0
        
        for pos, obj_type in mask_positions:
            if mask_index >= len(rle_data):
                break
                
            # If type changed or first mask, reset counter
            if obj_type != current_type:
                current_type = obj_type
                if obj_type:
                    type_counters[obj_type] = type_counters.get(obj_type, 0)
                else:
                    # Use generic "object" if can't infer type
                    obj_type = "object"
                    type_counters[obj_type] = type_counters.get(obj_type, 0)
            
            # Assign mask
            if obj_type:
                obj_id = type_counters[obj_type]
                mask_obj = Mask(obj_type, obj_id, mask_index, rle_data[mask_index])
                mask_key = f"{obj_type}_{obj_id}"
                mask_store[mask_key] = mask_obj
                type_counters[obj_type] = obj_id + 1
            else:
                mask_obj = Mask("object", mask_index, mask_index, rle_data[mask_index])
                mask_key = f"object_{mask_index}"
                mask_store[mask_key] = mask_obj
            
            mask_index += 1
        
        # Fill remaining masks if any
        while mask_index < len(rle_data):
            mask_obj = Mask("object", mask_index, mask_index, rle_data[mask_index])
            mask_key = f"object_{mask_index}"
            mask_store[mask_key] = mask_obj
            mask_index += 1
    
    # Fallback: if no masks found but we have RLE data, create generic masks
    elif len(rle_data) > 0:
        for region_id in range(len(rle_data)):
            mask_obj = Mask("object", region_id, region_id, rle_data[region_id])
            mask_key = f"object_{region_id}"
            mask_store[mask_key] = mask_obj

    return mask_store

if __name__ == "__main__":
    
    with open('../data/val/rephrased_val.json', 'r') as f:
        data = json.load(f)
    
    for item in tqdm(data[:100]):

        conversation = item['rephrase_conversations'][0]['value']
        normalized_answer = item['normalized_answer']
        rle_data = item['rle']

        mask_store = parse_masks_from_conversation(conversation, rle_data)

        if len(mask_store) != len(rle_data):
            print(f"Warning: Mismatch in mask count for item {item['id']}. "
                  f"Found {len(mask_store)} masks but expected {len(rle_data)}.")
            print(f"Conversations: {item['conversations'][0]['value']}")
            print(f"rephrase_conversations: {item['rephrase_conversations'][0]['value']}")
            import pdb; pdb.set_trace()

        print(conversation, normalized_answer)

        for key, mask in mask_store.items():
            print(f"{key}: {mask}")
        
        import pdb; pdb.set_trace()