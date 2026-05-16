import json
import os
from pathlib import Path

def main():
    data_dir = Path("/shared/ssd/home/b-s-adhikari/nn-gpt/VJEPA2/data/ssv2")
    video_dir = data_dir / "20bn-something-something-v2"
    
    if not video_dir.exists():
        print(f"Error: Video directory {video_dir} not found.")
        print("Please ensure the videos have been extracted.")
        return

    # Load labels
    labels_file = data_dir / "labels.json"
    if not labels_file.exists():
        print(f"Error: {labels_file} not found.")
        return
        
    with open(labels_file, "r") as f:
        labels_dict = json.load(f)
    
    # Create class name to integer ID mapping
    # labels.json in SSv2 is typically {"Pushing something from left to right": "1", ...}
    # We need 0-indexed integers
    class_mapping = {}
    for class_name, idx_str in labels_dict.items():
        class_mapping[class_name] = int(idx_str)

    # Process splits
    for split in ["train", "validation"]:
        json_file = data_dir / f"{split}.json"
        if not json_file.exists():
            print(f"Warning: {json_file} not found, skipping.")
            continue
            
        with open(json_file, "r") as f:
            split_data = json.load(f)
            
        csv_rows = []
        missing_count = 0
        
        for item in split_data:
            vid_id = item["id"]
            class_name = item["template"].replace("[", "").replace("]", "")
            label_idx = class_mapping[class_name]
            
            vid_path = video_dir / f"{vid_id}.webm"
            if vid_path.exists():
                csv_rows.append(f"{vid_path.resolve()} {label_idx}")
            else:
                missing_count += 1
                
        out_csv = data_dir / f"ssv2_{split}.csv"
        with open(out_csv, "w") as f:
            f.write("\n".join(csv_rows) + "\n")
            
        print(f"Wrote {len(csv_rows)} clips to {out_csv}")
        if missing_count > 0:
            print(f"Warning: {missing_count} videos were missing for {split} split.")

if __name__ == "__main__":
    main()
