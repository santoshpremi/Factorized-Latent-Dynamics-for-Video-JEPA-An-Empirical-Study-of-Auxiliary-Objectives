import os
from huggingface_hub import hf_hub_download

repo_id = "olarian/something-something-v2"
repo_type = "dataset"
local_dir = "/shared/ssd/home/b-s-adhikari/nn-gpt/VJEPA2/data/ssv2"

os.makedirs(local_dir, exist_ok=True)

files_to_download = [
    "labels.json",
    "train.json",
    "validation.json",
    "test.json",
]

for i in range(20):
    files_to_download.append(f"videos/20bn-something-something-v2-{i:02d}")

for file in files_to_download:
    print(f"Downloading {file}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=file,
        repo_type=repo_type,
        local_dir=local_dir,
        resume_download=True
    )

print("Download complete!")
