#!/usr/bin/env python3
"""
Debug script for the inside() function
"""

import json
import sys
import os
sys.path.append(os.path.dirname(__file__))

from tools import tools_api
from mask import parse_masks_from_conversation

def debug_specific_case(case_id="d09ebfffcdd52ec094434c4609403d57"):
    """Debug a specific error case"""
    
    # Load error case
    error_file = "../output/val_errors.json"
    with open(error_file, 'r') as f:
        errors = json.load(f)
    
    # Find the specific case
    case = None
    for err in errors:
        if err['id'] == case_id:
            case = err
            break
    
    if case is None:
        print(f"Case {case_id} not found!")
        return
    
    print(f"=== Debugging Case: {case_id} ===")
    print(f"Question: {case['question'][:200]}...")
    print(f"Predicted: {case['predicted']}")
    print(f"Ground Truth: {case['ground_truth']}")
    print(f"Category: {case['category']}")
    print()
    
    # Load the original data from val.json to get rle_data
    val_data_file = "../data/val.json"
    with open(val_data_file, 'r') as f:
        val_data = json.load(f)
    
    # Find the original data item
    original_item = None
    for item in val_data:
        if item['id'] == case_id:
            original_item = item
            break
    
    if original_item is None:
        print(f"Original data for {case_id} not found in val.json!")
        return
    
    # Parse masks from conversation using rle_data
    conversation = original_item['conversations'][0]['value']
    rle_data = original_item['rle']
    masks_dict = parse_masks_from_conversation(conversation, rle_data)
    
    print(f"Total masks found: {len(masks_dict)}")
    print(f"Mask keys: {list(masks_dict.keys())}")
    print()
    
    # Get image path
    # Extract image name from step_log
    img_name = None
    for step in case['step_log']:
        if step['step'] == 'set_question_start':
            img_name = step['data']['image']
            break
    
    if img_name is None:
        print("Image name not found in step_log!")
        return
    
    img_path = f"../data/val/images/{img_name}"
    print(f"Image path: {img_path}")
    
    # Initialize tools_api with different thresholds
    dist_model_cfg = {'model_path': '../distance_est/ckpt/epoch_5_iter_6831.pth'}
    
    # Use new dual-stream model with geometric features
    inside_model_cfg = {
        'model_path': '../inside_pred/ckpt_dual_stream/best_model.pth',
        'use_geometry': True,
        'input_channels': 5,
        'num_geo_features': 8
    }
    
    small_dist_model_cfg = {'model_path': '../distance_est/ckpt/3m_epoch6.pth'}
    
    # Test with different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for thresh in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing with inside_thres = {thresh}")
        print('='*60)
        
        tools = tools_api(
            dist_model_cfg=dist_model_cfg,
            inside_model_cfg=inside_model_cfg,
            small_dist_model_cfg=small_dist_model_cfg,
            inside_thres=thresh,
            img_path=img_path
        )
        
        tools.update_masks(list(masks_dict.values()))
        
        # Get all pallets and buffers
        pallets = [m for m in masks_dict.values() if m.object_class == 'pallet']
        buffers = [m for m in masks_dict.values() if m.object_class == 'buffer']
        
        print(f"Pallets: {[p.mask_name() for p in pallets]}")
        print(f"Buffers: {[b.mask_name() for b in buffers]}")
        print()
        
        # Test each buffer
        for buffer in buffers:
            count = tools.inside(buffer, pallets, debug=True)
            print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_id', type=str, default='664dce6f776054867e5277f9daab413b',
                      help='Case ID to debug')
    args = parser.parse_args()
    
    debug_specific_case(args.case_id)
