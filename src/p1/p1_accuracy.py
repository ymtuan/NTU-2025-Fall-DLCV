import json
import sys
from collections import defaultdict

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
    
    # Create a mapping of question_id to ground truth
    gt_map = {}
    for i, item in enumerate(annotations):
        answer = item['answer'].lower().strip()
        # Normalize answer to "Yes" or "No"
        gt_answer = "Yes" if answer in ["yes", "y"] else "No"
        gt_map[i] = gt_answer
    
    print(f"Total annotations: {len(gt_map)}")
    print(f"Total predictions: {len(predictions)}")
    
    # Calculate accuracy
    correct = 0
    total = 0
    
    for pred in predictions:
        question_id = pred['question_id']
        predicted_answer = pred['text'].strip()
        
        if question_id not in gt_map:
            print(f"Warning: question_id {question_id} not found in annotations")
            continue
        
        ground_truth = gt_map[question_id]
        
        # Normalize prediction to "Yes" or "No"
        pred_normalized = "Yes" if "yes" in predicted_answer.lower() else "No"
        
        if pred_normalized == ground_truth:
            correct += 1
        
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"Correct: {correct}/{total}")
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