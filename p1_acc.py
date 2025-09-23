import sys
import pandas as pd

def compare_csv(pred_csv, gt_csv):
    # Load CSVs
    df_pred = pd.read_csv(pred_csv)
    df_gt = pd.read_csv(gt_csv)

    # Check both have same length
    if len(df_pred) != len(df_gt):
        print("Warning: CSVs have different number of rows!")

    # Compare labels
    correct = (df_pred['label'] == df_gt['label']).sum()
    total = len(df_gt)
    accuracy = correct / total

    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    # Optional: show mismatches
    mismatches = df_pred[df_pred['label'] != df_gt['label']]
    if len(mismatches) > 0:
        print("\nMismatched rows:")
        print(mismatches)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 compare_csv.py <pred_csv> <ground_truth_csv>")
    else:
        compare_csv(sys.argv[1], sys.argv[2])
