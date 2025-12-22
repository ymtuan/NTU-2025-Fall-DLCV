#!/usr/bin/env python3
"""
Analyze val_errors.json to understand error patterns
"""
import json
import argparse
from collections import defaultdict
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='output/val_errors.json',
                        help='Path to val_errors.json')
    return parser.parse_args()

def analyze_errors(errors):
    """Analyze error patterns from val_errors.json"""
    
    # Category breakdown
    category_errors = defaultdict(list)
    for e in errors:
        cat = e.get('category') or e.get('ground_truth_category', 'unknown')
        category_errors[cat].append(e)
    
    print("=" * 80)
    print("VAL SET ERROR ANALYSIS")
    print("=" * 80)
    print(f"\n📊 Total Errors: {len(errors)}\n")
    
    # Category summary
    print("📈 Errors by Category:")
    for cat, errs in sorted(category_errors.items(), key=lambda x: -len(x[1])):
        print(f"   {cat}: {len(errs)} errors")
    
    # Analyze each category
    for cat in ['count', 'distance', 'mcq', 'left_right']:
        if cat not in category_errors:
            continue
        errs = category_errors[cat]
        print(f"\n{'─' * 60}")
        print(f"🔴 {cat.upper()} ERRORS: {len(errs)}")
        print(f"{'─' * 60}")
        
        if cat == 'count':
            analyze_count_errors(errs)
        elif cat == 'distance':
            analyze_distance_errors(errs)
        elif cat == 'mcq':
            analyze_mcq_errors(errs)
        elif cat == 'left_right':
            analyze_left_right_errors(errs)
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print_recommendations(category_errors)

def analyze_count_errors(errors):
    """Analyze count category errors"""
    off_by_1 = 0
    off_by_2_plus = 0
    over_count = 0
    under_count = 0
    
    question_patterns = defaultdict(int)
    
    for e in errors:
        try:
            pred = int(e['predicted'])
            gt = int(e['ground_truth'])
            diff = pred - gt
            
            if abs(diff) == 1:
                off_by_1 += 1
            else:
                off_by_2_plus += 1
            
            if diff > 0:
                over_count += 1
            elif diff < 0:
                under_count += 1
            
            # Analyze question patterns
            q = e.get('question', '').lower()
            if 'left' in q and 'buffer' in q:
                question_patterns['left buffer'] += 1
            elif 'right' in q and 'buffer' in q:
                question_patterns['right buffer'] += 1
            elif 'middle' in q:
                question_patterns['middle'] += 1
            else:
                question_patterns['other'] += 1
                
        except (ValueError, TypeError):
            pass
    
    print(f"\n   ±1 errors: {off_by_1}")
    print(f"   ±2+ errors: {off_by_2_plus}")
    print(f"\n   Over-counting: {over_count}")
    print(f"   Under-counting: {under_count}")
    
    print(f"\n   🔍 Question patterns:")
    for pattern, count in sorted(question_patterns.items(), key=lambda x: -x[1]):
        print(f"      {pattern}: {count}")
    
    # Sample errors
    print(f"\n   📝 Sample errors (first 3):")
    for e in errors[:3]:
        q = e.get('question', '')[:80]
        print(f"      Q: {q}...")
        print(f"      Pred: {e['predicted']}, GT: {e['ground_truth']}")
        print()

def analyze_distance_errors(errors):
    """Analyze distance category errors"""
    gt_zero_errors = []
    underestimate = []
    overestimate = []
    
    for e in errors:
        try:
            pred_str = e['predicted']
            gt_str = e['ground_truth']
            
            # Handle special cases
            if pred_str in ['-1', 'error', 'None']:
                continue
                
            pred = float(pred_str)
            gt = float(gt_str)
            
            if gt == 0 and pred != 0:
                gt_zero_errors.append((pred, e))
            elif pred < gt:
                underestimate.append((pred, gt, e))
            elif pred > gt:
                overestimate.append((pred, gt, e))
        except (ValueError, TypeError):
            pass
    
    print(f"\n   🎯 GT=0 but predicted non-zero: {len(gt_zero_errors)}")
    if gt_zero_errors:
        print("      Sample predictions:")
        for pred, e in gt_zero_errors[:5]:
            q = e.get('question', '')
            snippet = q[50:100] if len(q) > 50 else q
            print(f"      - pred={pred:.2f}, snippet: ...{snippet}...")
    
    print(f"\n   📉 Underestimate (pred < gt): {len(underestimate)}")
    if underestimate:
        # Sort by error magnitude
        underestimate.sort(key=lambda x: (x[1] - x[0]) / max(x[1], 0.01), reverse=True)
        print("      Top 5 worst:")
        for pred, gt, e in underestimate[:5]:
            pct = abs(pred - gt) / max(gt, 0.01) * 100
            print(f"      - pred={pred:.2f}, gt={gt:.2f} (off by {pct:.1f}%)")
    
    print(f"\n   📈 Overestimate (pred > gt): {len(overestimate)}")
    if overestimate:
        overestimate.sort(key=lambda x: (x[0] - x[1]) / max(x[1], 0.01), reverse=True)
        print("      Top 5 worst:")
        for pred, gt, e in overestimate[:5]:
            pct = abs(pred - gt) / max(gt, 0.01) * 100
            print(f"      - pred={pred:.2f}, gt={gt:.2f} (off by {pct:.1f}%)")

def analyze_mcq_errors(errors):
    """Analyze MCQ category errors"""
    print(f"\n   Total MCQ errors: {len(errors)}")
    
    # Categorize by question type
    patterns = defaultdict(list)
    for e in errors:
        q = e.get('question', '').lower()
        if 'transporter' in q and 'pallet' in q:
            patterns['transporter-pallet selection'].append(e)
        elif 'empty' in q or 'idle' in q:
            patterns['empty transporter'].append(e)
        elif 'closest' in q:
            patterns['closest selection'].append(e)
        else:
            patterns['other'].append(e)
    
    print(f"\n   🔍 Error patterns:")
    for pattern, errs in sorted(patterns.items(), key=lambda x: -len(x[1])):
        print(f"      {pattern}: {len(errs)}")
    
    print(f"\n   📝 Sample errors (first 3):")
    for e in errors[:3]:
        q = e.get('question', '')[:80]
        print(f"      Q: {q}...")
        print(f"      Pred: {e['predicted']}, GT: {e['ground_truth']}")
        print()

def analyze_left_right_errors(errors):
    """Analyze left_right category errors"""
    print(f"\n   Total left_right errors: {len(errors)}")
    
    gt_left_pred_right = 0
    gt_right_pred_left = 0
    
    for e in errors:
        pred = e['predicted'].lower().strip()
        gt = e['ground_truth'].lower().strip()
        
        if gt == 'left' and pred == 'right':
            gt_left_pred_right += 1
        elif gt == 'right' and pred == 'left':
            gt_right_pred_left += 1
    
    print(f"      GT=left, Pred=right: {gt_left_pred_right}")
    print(f"      GT=right, Pred=left: {gt_right_pred_left}")
    
    print(f"\n   📝 Sample errors (first 3):")
    for e in errors[:3]:
        q = e.get('question', '')[:80]
        print(f"      Q: {q}...")
        print(f"      Pred: {e['predicted']}, GT: {e['ground_truth']}")
        
        # Try to find the answer in conversation
        conv = e.get('conversation', [])
        for msg in reversed(conv):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if 'left' in content.lower() or 'right' in content.lower():
                    answer_snippet = content[:100]
                    print(f"      Answer: {answer_snippet}...")
                    break
        print()

def print_recommendations(category_errors):
    """Print recommendations based on error analysis"""
    
    if 'count' in category_errors:
        errs = category_errors['count']
        over = sum(1 for e in errs if int(e.get('predicted', 0)) > int(e.get('ground_truth', 0)))
        under = len(errs) - over
        print(f"\n1. COUNT ({len(errs)} errors):")
        if over > under:
            print(f"   - Over-counting ({over}) > Under-counting ({under})")
            print(f"   - Consider RAISING inside_thres to be more strict")
        else:
            print(f"   - Under-counting ({under}) > Over-counting ({over})")
            print(f"   - Consider LOWERING inside_thres to be more lenient")
    
    if 'distance' in category_errors:
        errs = category_errors['distance']
        gt_zero = sum(1 for e in errs if e.get('ground_truth') == '0' or e.get('ground_truth') == 0)
        print(f"\n2. DISTANCE ({len(errs)} errors):")
        if gt_zero > 0:
            print(f"   - {gt_zero} errors where GT=0 (overlapping objects)")
            print(f"   - Check IoU/intersection thresholds in dist() function")
    
    if 'mcq' in category_errors:
        errs = category_errors['mcq']
        print(f"\n3. MCQ ({len(errs)} errors):")
        print(f"   - Consider using dedicated closest_pred model")
        print(f"   - Check if empty transporter detection is working")
    
    if 'left_right' in category_errors:
        errs = category_errors['left_right']
        print(f"\n4. LEFT_RIGHT ({len(errs)} errors):")
        if len(errs) > 5:
            print(f"   - Check _convert_boolean_answer_to_direction_fallback logic")
            print(f"   - Ensure is_left semantics: True -> left, False -> right")

def main():
    args = parse_args()
    
    with open(args.input, 'r') as f:
        errors = json.load(f)
    
    analyze_errors(errors)

if __name__ == '__main__':
    main()
