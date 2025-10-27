python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

model_id = "liuhaotian/llava-v1.5-7b"
local_dir = os.path.expanduser("~/dl/dlcv-fall-2025-hw3-ymtuan/llava-v1.5-7b")

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
print("Done!")
EOF
