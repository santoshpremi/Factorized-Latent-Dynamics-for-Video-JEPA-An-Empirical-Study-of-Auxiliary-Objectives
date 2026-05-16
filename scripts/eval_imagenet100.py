import os
import sys
import torch
import torch.nn as nn
import yaml
import json
import argparse
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Add vjepa2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../vjepa2'))
from app.vjepa_2_1.utils import init_video_model

def load_model(config_path, checkpoint_path, device):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model_name = cfg['model']['model_name']
    patch_size = cfg['data']['patch_size']
    crop_size = cfg['data']['crop_size']
    tubelet_size = cfg['data']['tubelet_size']
    num_frames = cfg['data']['dataset_fpcs'][0]
    
    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=8,
        use_rope=cfg['model'].get('use_rope', False),
        modality_embedding=cfg['model'].get('modality_embedding', False),
        interpolate_rope=cfg['model'].get('interpolate_rope', False),
        use_sdpa=cfg['meta'].get('use_sdpa', False),
        uniform_power=cfg['model'].get('uniform_power', False),
        use_mask_tokens=cfg['model'].get('use_mask_tokens', False),
        zero_init_mask_tokens=cfg['model'].get('zero_init_mask_tokens', False),
        has_cls_first=cfg['model'].get('has_cls_first', False),
        img_temporal_dim_size=cfg['model'].get('img_temporal_dim_size', None),
        n_registers=cfg['model'].get('n_registers', 0),
        n_registers_predictor=cfg['model'].get('n_registers_predictor', 0),
    )
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    encoder_state_dict = {}
    for k, v in checkpoint['target_encoder'].items():
        new_k = k.replace('module.', '')
        encoder_state_dict[new_k] = v
        
    encoder.load_state_dict(encoder_state_dict, strict=False)
    encoder.to(device)
    encoder.eval()
    
    return encoder, cfg, num_frames

def extract_features(dataloader, encoder, device, num_frames):
    features = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            # images shape: [B, C, H, W]
            B, C, H, W = images.shape
            
            # To feed into video encoder, we need [B, C, T, H, W]
            # We repeat the static image across the temporal dimension
            videos = images.unsqueeze(2).repeat(1, 1, num_frames, 1, 1).to(device)
            
            # Extract features
            # Note: encoder expects a list of tensors for different resolutions, we pass a list with one tensor
            h = encoder([videos], gram_mode=False, training_mode=False)
            
            # h[-1] shape: [B, N_patches, embed_dim]
            z = h[-1]
            
            # Global Average Pooling for static images
            z_pooled = z.mean(dim=1).cpu().half() # [B, embed_dim]
            
            features.append(z_pooled)
            labels_list.append(labels)
            
    return torch.cat(features, dim=0), torch.cat(labels_list, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_json', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    encoder, cfg, num_frames = load_model(args.config, args.checkpoint, device)
    crop_size = cfg['data']['crop_size']
    
    base_dir = '/a/mm/VJEPA2/data/imagenet100'
    print(f"Loading ImageNet-100 data from {base_dir}...")
    
    transform = T.Compose([
        T.Resize(crop_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(os.path.join(base_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(os.path.join(base_dir, 'validation'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    print("Extracting FULL training features...")
    X_train, y_train = extract_features(train_loader, encoder, device, num_frames)
    
    print("Extracting FULL validation features...")
    X_test, y_test = extract_features(val_loader, encoder, device, num_frames)
    
    print(f"Train Features: {X_train.shape}, Test Features: {X_test.shape}")
    
    # Free the encoder from GPU memory
    del encoder
    torch.cuda.empty_cache()
    
    print(f"Training Linear Probe (100 classes) for {args.epochs} epochs...")
    # Simple Linear Probe
    probe = nn.Linear(X_train.shape[-1], 100).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 256
    
    for epoch in range(args.epochs):
        probe.train()
        permutation = torch.randperm(X_train.size()[0])
        total_loss = 0
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices].to(device).float(), y_train[indices].to(device)
            
            optimizer.zero_grad()
            outputs = probe(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/(X_train.size()[0]/batch_size):.4f}")
            
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, X_test.size()[0], batch_size):
            batch_x, batch_y = X_test[i:i+batch_size].to(device).float(), y_test[i:i+batch_size].to(device)
            outputs = probe(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).float().sum().item()
            total += batch_y.size(0)
            
    acc = correct / total
    
    print(f"\n========================================")
    print(f"[{args.name}] ImageNet-100 Classification Accuracy: {acc*100:.2f}%")
    print(f"========================================")

    results = {
        "name": args.name,
        "dataset": "ImageNet-100",
        "task": "frozen-encoder + linear probe (100-class)",
        "train_probe_samples": len(X_train),
        "val_samples": len(X_test),
        "top1_accuracy_percent": round(acc * 100, 2),
        "params": {"lr": args.lr, "weight_decay": args.wd, "epochs": args.epochs},
        "checkpoint": args.checkpoint,
        "config": args.config
    }
    
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.out_json}")

if __name__ == '__main__':
    main()
