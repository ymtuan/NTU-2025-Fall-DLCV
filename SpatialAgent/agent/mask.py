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

def parse_masks_from_conversation(conversation: str, rle_data: List[Dict], llm_client=None) -> Dict[str, Mask]:
    """
    Parses mask references from the conversation and builds Mask objects,
    storing them in a dictionary with keys like 'pallet_mask_0'.
    
    Supports two formats:
    1. <object_class>_<id> format (e.g., <pallet_0>, <buffer_1>) - used in test set
    2. <mask> format - used in train set, masks are assigned sequentially
    
    Args:
        conversation: The conversation/question text containing mask references
        rle_data: List of RLE dictionaries for the masks
        llm_client: Optional LLM client for mask type classification (if None, uses rule-based)
    """
    # Regex to find masks: matches <pallet_mask_0>, <transporter_mask_1> etc.
    mask_pattern = re.compile(r"<([a-zA-Z]+)_(\d+)>")
    
    # Regex to find generic <mask> tags
    # Use same pattern as in classification functions to ensure consistency
    generic_mask_pattern = re.compile(r"<mask(?:>)?")

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
        # Use LLM for mask type classification if available
        if llm_client is not None:
            print(f"Using LLM for mask type classification with {len(generic_masks)} masks")
            mask_positions = _classify_masks_with_llm(conversation, generic_masks, llm_client)
            print(mask_positions)
        else:
            # Fallback to rule-based classification
            print(f"Using rule-based classification with {len(generic_masks)} masks")
            mask_positions = _classify_masks_rule_based(conversation, generic_masks)
            print(mask_positions)
        
        # Validate: Ensure mask_positions count matches rle_data count
        if len(mask_positions) != len(rle_data):
            print(f"Warning: mask_positions count ({len(mask_positions)}) != rle_data count ({len(rle_data)})")
            # Truncate or pad as needed
            if len(mask_positions) > len(rle_data):
                mask_positions = mask_positions[:len(rle_data)]
            else:
                # Pad with 'object' type for remaining positions
                last_pos = mask_positions[-1][0] if mask_positions else 0
                for i in range(len(mask_positions), len(rle_data)):
                    mask_positions.append((last_pos + i * 10, 'object'))
        
        # Group consecutive masks by object type and create Mask objects
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


import re
import json
from typing import List, Tuple

def _classify_masks_with_llm(conversation: str, generic_masks: List, llm_client) -> List[Tuple[int, str]]:
    """
    Use LLM to classify mask types by rewriting the sentence with explicit tags.
    Strategy: Ask LLM to replace <mask> with <pallet>, <buffer>, <shelf>, <transporter>.
    """
    mask_count = len(generic_masks)
    
    # Few-shot examples are POWERFUL. Don't explain rules, show them.
    prompt = f"""You are a data parser. Your task is to replace the generic `<mask>` tags in the input text with specific object type tags based on the context.
    
    Allowed tags: `<pallet>`, `<buffer>`, `<transporter>`, `<shelf>`, `<object>` (if unsure).
    
    Examples:
    Input: "Given the pallets <mask> <mask> and transporter <mask>, which is closest?"
    Output: "Given the pallets <pallet> <pallet> and transporter <transporter>, which is closest?"
    
    Input: "Count the items in the buffer region among <mask> <mask> <mask>"
    Output: "Count the items in the buffer region among <buffer> <buffer> <buffer>"
    
    Input: "What is the distance between the rightmost shelf among <mask> <mask> and the pallet <mask>?"
    Output: "What is the distance between the rightmost shelf among <shelf> <shelf> and the pallet <pallet>?"
    
    Input: "transporter masks <mask> <mask <mask and buffer masks <mask <mask"
    Output: "transporter masks <transporter> <transporter> <transporter> and buffer masks <buffer> <buffer>"

    Input: "{conversation}"
    Output:"""

    try:
        # Call LLM
        # Note: temperature is set during client initialization, not here
        response = llm_client.send_message(prompt)
        
        # Clean response
        response = response.strip()
        if response.startswith("Output:"):
            response = response[7:].strip()
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
            
        # Parse the new tags
        # Find all tags like <type>
        tag_pattern = re.compile(r"<(pallet|buffer|transporter|shelf|object)>", re.IGNORECASE)
        found_tags = tag_pattern.findall(response)
        
        # 1. Validation: Check Count
        if len(found_tags) != mask_count:
            print(f"Warning: LLM returned {len(found_tags)} tags, expected {mask_count}. Falling back.")
            # Optional: Retry logic could go here
            return _classify_masks_rule_based(conversation, generic_masks)
            
        # 2. Map back to original positions
        # We use the original generic_mask_pattern to get start indices
        generic_mask_pattern = re.compile(r"<mask(?:>)?") # Handle potential typo <mask
        original_matches = list(generic_mask_pattern.finditer(conversation))
        
        mask_positions = []
        for i, match in enumerate(original_matches):
            obj_type = found_tags[i].lower()
            mask_positions.append((match.start(), obj_type))
            
        return mask_positions

    except Exception as e:
        print(f"LLM Classification Error: {e}")
        return _classify_masks_rule_based(conversation, generic_masks)

def _classify_masks_rule_based(conversation: str, generic_masks: List) -> List[Tuple[int, str]]:
    """
    Robust fallback logic.
    Idea: Split sentence by object keywords and assign types to following masks.
    """
    generic_mask_pattern = re.compile(r"<mask(?:>)?")
    mask_positions = []
    
    # Default type
    current_type = 'object'
    
    # Simple state machine scan
    # Tokenize by identifying keywords and masks
    # This is a simplified version; a full implementation would iterate through string indices
    
    # Let's map positions by finding closest preceding keyword
    for match in generic_mask_pattern.finditer(conversation):
        start_idx = match.start()
        
        # Look backwards from this mask to find the 'governing' noun
        preceding_text = conversation[:start_idx].lower()
        
        # Define keywords and their search priority (right-to-left)
        keywords = {
            'pallet': preceding_text.rfind('pallet'),
            'buffer': preceding_text.rfind('buffer'),
            'transporter': preceding_text.rfind('transporter'),
            'shelf': preceding_text.rfind('shelf')
        }
        
        # Filter out not found (-1)
        found_keywords = {k: v for k, v in keywords.items() if v != -1}
        
        if not found_keywords:
            obj_type = 'object' # No keyword found before this mask
        else:
            # Find the keyword with the largest index (closest to the mask)
            closest_type = max(found_keywords, key=found_keywords.get)
            
            # Distance check (optional): if keyword is too far, maybe reset?
            # For now, we assume the type persists until a new keyword appears.
            obj_type = closest_type
            
        mask_positions.append((start_idx, obj_type))
    
    # Validate: Ensure we found the same number of masks as expected
    expected_count = len(generic_masks)
    if len(mask_positions) != expected_count:
        print(f"Warning: Rule-based classification found {len(mask_positions)} masks, expected {expected_count}")
    
    return mask_positions

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