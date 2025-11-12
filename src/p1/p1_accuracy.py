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
    """
    
    # Load predictions (support array JSON and NDJSON)
    print(f"Loading predictions from {pred_file}...")
    predictions = []
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                predictions = json.load(f)
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            predictions.append(json.loads(line))
                        except Exception as e:
                            print(f"Warning: skip unparsable line. Error: {e}")
    except Exception as e:
        print(f"Error reading predictions: {e}")
        sys.exit(1)
    
    # Load ground truth annotations
    print(f"Loading annotations from {annotation_file}...")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Build GT maps: by question_id and by (image_source, question)
    gt_by_qid = {}
    gt_by_pair = {}
    for i, item in enumerate(annotations):
        qid = item.get("question_id", i)
        gt_ans = _normalize_yes_no(item.get("answer", ""))
        if gt_ans is None:
            gt_ans = "No"
        gt_by_qid[qid] = gt_ans
        key = (item.get("image_source"), item.get("question"))
        gt_by_pair[key] = gt_ans
    
    print(f"Total annotations: {len(annotations)}")
    print(f"Total predictions: {len(predictions)}")
    
    # Calculate accuracy with flexible matching
    correct = 0
    total = 0
    ambiguous = 0
    unmatched = 0
    for pred in predictions:
        # Normalize predicted answer
        predicted_answer_raw = pred.get('predict', '')
        pred_normalized = _normalize_yes_no(predicted_answer_raw)
        if pred_normalized is None:
            ambiguous += 1
            pred_normalized = "No"
        
        # Match strategy: question_id -> (image_source, question)
        matched = False
        ground_truth = None
        
        qid = pred.get('question_id')
        if qid is not None and qid in gt_by_qid:
            ground_truth = gt_by_qid[qid]
            matched = True
        else:
            key = (pred.get('image_source'), pred.get('question'))
            if key in gt_by_pair:
                ground_truth = gt_by_pair[key]
                matched = True
        
        if not matched:
            print(f"Warning: could not match prediction to annotation for image_source={pred.get('image_source')} question={pred.get('question')}")
            unmatched += 1
            continue
        
        if pred_normalized == ground_truth:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    print(f"Matched: {total} | Unmatched: {unmatched}")
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