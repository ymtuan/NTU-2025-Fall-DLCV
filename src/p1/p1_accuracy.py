import json
import sys
from collections import defaultdict
import re

def _normalize_yes_no(text):
    if text is None:
        return None
    t = text.strip().lower()
    # Take first alphanumeric token
    tokens = re.findall(r"[a-zA-Z]+", t)
    if not tokens:
        return None
    first = tokens[0]
    if first in ("yes", "y"):
        return "Yes"
    if first in ("no", "n"):
        return "No"
    return None

def evaluate_accuracy(pred_file, annotation_file):
    """
    Evaluate accuracy on POPE dataset
    
    Args:
        pred_file: path to predictions JSON (output from inference)
        annotation_file: path to ground truth JSON
    
    Returns:
        accuracy: float between 0 and 1
    """
    
    # Load predictions
    print(f"Loading predictions from {pred_file}...")
    with open(pred_file) as f:
        predictions = json.load(f)
    
    # Load ground truth annotations
    print(f"Loading annotations from {annotation_file}...")
    with open(annotation_file) as f:
        annotations = json.load(f)
    
    # Build gt_map using provided question_id if exists
    gt_map = {}
    for i, item in enumerate(annotations):
        qid = item.get("question_id", i)
        gt_ans = _normalize_yes_no(item.get("answer", ""))
        if gt_ans is None:
            # Fallback: treat anything not recognized as "No"
            gt_ans = "No"
        gt_map[qid] = gt_ans
    
    print(f"Total annotations: {len(gt_map)}")
    print(f"Total predictions: {len(predictions)}")
    
    # Calculate accuracy
    correct = 0
    total = 0
    ambiguous = 0
    for pred in predictions:
        question_id = pred.get('question_id')
        predicted_answer_raw = pred.get('text', '')
        
        if question_id not in gt_map:
            print(f"Warning: question_id {question_id} not found in annotations")
            continue
        
        ground_truth = gt_map[question_id]
        
        pred_normalized = _normalize_yes_no(predicted_answer_raw)
        if pred_normalized is None:
            ambiguous += 1
            pred_normalized = "No"  # default instead of substring heuristic
        if pred_normalized == ground_truth:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    print(f"Correct: {correct}/{total}")
    print(f"Ambiguous predictions (fallback applied): {ambiguous}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{'='*50}\n")
    
    return accuracy

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate_accuracy.py <pred_file> <annotation_file>")
        print("Example: python3 evaluate_accuracy.py output/p1_1_pred.json ../../hw3_data/p1_data/val.json")
        sys.exit(1)
    
    pred_file = sys.argv[1]
    annotation_file = sys.argv[2]
    
    accuracy = evaluate_accuracy(pred_file, annotation_file)