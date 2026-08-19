"""
Upload ONNX weights to the Teamrat/habit HuggingFace model repository.

Run this once after converting weights:
    python upload_onnx_to_hf.py

Requires: huggingface_hub and a valid HF token (huggingface-cli login).
"""

import os
from huggingface_hub import HfApi

REPO_ID = "Teamrat/habit"
ONNX_DIR = os.path.join(os.path.dirname(__file__), "onnx_weights")

api = HfApi()

for i in range(1, 21):
    name = f"member_{i:02d}.onnx"
    local_path = os.path.join(ONNX_DIR, name)
    if not os.path.exists(local_path):
        print(f"  SKIP {name} — not found")
        continue

    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Uploading {name} ({size_mb:.1f} MB)...", end=" ", flush=True)

    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"onnx/{name}",
        repo_id=REPO_ID,
        repo_type="model",
    )
    print("done")

print(f"\nAll ONNX weights uploaded to https://huggingface.co/{REPO_ID}")
