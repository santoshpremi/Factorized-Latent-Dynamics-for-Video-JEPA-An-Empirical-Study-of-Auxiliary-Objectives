import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def download_imagenet100(output_dir):
    print("Downloading ImageNet-100 from Hugging Face...")
    # Load the dataset
    dataset = load_dataset("clane9/imagenet-100")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train and validation splits
    for split in ['train', 'validation']:
        print(f"Processing {split} split...")
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        split_data = dataset[split]
        for i, item in enumerate(tqdm(split_data)):
            image = item['image']
            label = item['label']
            
            # Create class directory
            class_dir = os.path.join(split_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            
            # Save image
            img_path = os.path.join(class_dir, f"{split}_{i}.jpg")
            
            # Convert to RGB if it's not (e.g. grayscale or RGBA)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            image.save(img_path)
            
    print("Download and extraction complete!")

if __name__ == "__main__":
    output_dir = "/shared/ssd/home/b-s-adhikari/nn-gpt/VJEPA2/data/imagenet100"
    download_imagenet100(output_dir)
