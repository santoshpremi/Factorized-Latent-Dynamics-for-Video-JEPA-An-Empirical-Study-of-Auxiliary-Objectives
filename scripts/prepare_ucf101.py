import os
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
    data_dir = Path("/a/mm/VJEPA2/data/ucf101")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading UCF-101.zip from HuggingFace...")
    videos_zip = hf_hub_download(
        repo_id="quchenyuan/UCF101-ZIP", 
        filename="UCF-101.zip", 
        repo_type="dataset",
        local_dir=str(data_dir)
    )
    
    print("Downloading splits from HuggingFace...")
    splits_zip = hf_hub_download(
        repo_id="quchenyuan/UCF101-ZIP", 
        filename="UCF101TrainTestSplits-RecognitionTask.zip", 
        repo_type="dataset",
        local_dir=str(data_dir)
    )
    
    print("Extracting videos...")
    with zipfile.ZipFile(videos_zip, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    print("Extracting splits...")
    with zipfile.ZipFile(splits_zip, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    # The videos are extracted to data_dir / "UCF-101"
    video_root = data_dir / "UCF-101"
    split_root = data_dir / "ucfTrainTestlist"
    
    # Read class indices
    class_mapping = {}
    with open(split_root / "classInd.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                idx, class_name = parts
                # 0-indexed labels for PyTorch
                class_mapping[class_name] = int(idx) - 1
                
    # Process train/val splits
    for split_name, txt_file, out_csv in [
        ("train", "trainlist01.txt", "ucf101_train.csv"),
        ("val", "testlist01.txt", "ucf101_val.csv")
    ]:
        csv_rows = []
        with open(split_root / txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                
                # trainlist has "path label", testlist has just "path"
                rel_path = parts[0]
                class_name = rel_path.split('/')[0]
                label = class_mapping[class_name]
                
                abs_path = video_root / rel_path
                if abs_path.exists():
                    csv_rows.append(f"{abs_path.resolve()} {label}")
                else:
                    print(f"Warning: {abs_path} not found")
                    
        out_path = data_dir / out_csv
        out_path.write_text("\n".join(csv_rows) + "\n")
        print(f"Wrote {len(csv_rows)} clips to {out_path}")
        
    print("Done! UCF-101 is ready for V-JEPA2.")

if __name__ == "__main__":
    main()
