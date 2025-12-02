import json

INPUT_FILE = "../../hw3_data/p2_data/val.json"
OUTPUT_FILE = "../../hw3_data/p2_data/short_val.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

# keep only first 250 annotations
data["annotations"] = data["annotations"][:250]

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved {len(data['annotations'])} annotations to {OUTPUT_FILE}")
