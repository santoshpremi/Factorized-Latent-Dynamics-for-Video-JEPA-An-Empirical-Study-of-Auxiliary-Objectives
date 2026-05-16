import os
import sys
from pathlib import Path
from datasets import load_dataset
import torchvision.io as tvio
import torch

out_dir = Path("/shared/ssd/home/b-s-adhikari/nn-gpt/VJEPA2/data/ucf101")
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading UCF101 from Hugging Face...")
# The HF dataset provides video paths if we just download it
ds = load_dataset("ucf101", split="train")

print(f"Loaded {len(ds)} videos. Saving to {out_dir}...")
# It might already be downloaded to ~/.cache/huggingface/datasets
# Let's just write a CSV with absolute paths
rows = []
for i, item in enumerate(ds):
    # video is a string path or a dict
    vid_path = item["video"]
    label = item["label"]
    rows.append(f"{vid_path} {label}")

csv_path = out_dir / "ucf101_train.csv"
csv_path.write_text("\n".join(rows) + "\n")
print(f"Wrote CSV to {csv_path}")
