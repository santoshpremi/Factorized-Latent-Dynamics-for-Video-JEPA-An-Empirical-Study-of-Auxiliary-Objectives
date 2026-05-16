import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T
import torch.nn.functional as F

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

def load_video(path, num_frames, crop_size):
    import decord
    decord.bridge.set_bridge('torch')
    try:
        vr = decord.VideoReader(path, num_threads=1)
        total_frames = len(vr)
        
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = np.pad(np.arange(total_frames), (0, num_frames - total_frames), mode='edge')
            
        v = vr.get_batch(indices) # [T, H, W, C]
        v = v.permute(3, 0, 1, 2).float() / 255.0 # [C, T, H, W]
        
        _, T, H, W = v.shape
        scale = crop_size / min(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        v = F.interpolate(v, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        top = (new_H - crop_size) // 2
        left = (new_W - crop_size) // 2
        v = v[:, :, top:top+crop_size, left:left+crop_size]
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        v = (v - mean) / std
        return v
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

class AttentiveProbe(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_attn, _ = self.attn(x, x, x)
        x = x + x_attn
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

def extract_features(csv_path, encoder, device, num_frames, crop_size):
    # Read CSV
    data_list = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                data_list.append({'path': parts[0], 'label': int(parts[1])})
                
    print(f"Loaded {len(data_list)} items from {csv_path}")

    # Pre-allocate memory to prevent OOM errors
    features = None
    labels = torch.zeros(len(data_list), dtype=torch.long)
    
    valid_idx = 0
    with torch.no_grad():
        for i, item in enumerate(tqdm(data_list)):
            vid_path = item['path']
            vid_path = vid_path.replace('/shared/ssd/home/b-s-adhikari/nn-gpt', '/a/mm')
            v = load_video(vid_path, num_frames, crop_size)
            if v is None:
                continue
            v = v.unsqueeze(0).to(device)
            
            h = encoder([v], gram_mode=False, training_mode=False)
            z = h[-1].squeeze(0).cpu().half()
            
            # Average pool tokens to get a single vector per video to prevent OOM
            # z shape is [num_tokens, embed_dim] -> [embed_dim]
            z = z.mean(dim=0)
            
            if features is None:
                features = torch.zeros((len(data_list), *z.shape), dtype=torch.float16)
                
            features[valid_idx] = z
            labels[valid_idx] = item['label']
            valid_idx += 1
            
    if features is None:
        print("Error: No valid features extracted. Check video paths.")
        sys.exit(1)
        
    return features[:valid_idx], labels[:valid_idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_json', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    encoder, cfg, num_frames = load_model(args.config, args.checkpoint, device)
    crop_size = cfg['data']['crop_size']
    
    base_dir = '/a/mm/VJEPA2/data/ssv2'
    train_csv = os.path.join(base_dir, 'ssv2_train.csv')
    val_csv = os.path.join(base_dir, 'ssv2_validation.csv')
    
    print(f"Loading SSv2 data from {base_dir}...")
    
    print("Extracting FULL training features...")
    X_train, y_train = extract_features(train_csv, encoder, device, num_frames, crop_size)
    
    print("Extracting FULL validation features...")
    X_test, y_test = extract_features(val_csv, encoder, device, num_frames, crop_size)
    
    print(f"Train Features: {X_train.shape}, Test Features: {X_test.shape}")
    
    # Free the encoder from GPU memory to make room for the probe
    del encoder
    torch.cuda.empty_cache()
    
    # SSv2 has 174 classes
    num_classes = 174
    print(f"Training Attentive Probe ({num_classes} classes) for {args.epochs} epochs...")
    probe = AttentiveProbe(embed_dim=X_train.shape[-1], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 16
    
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
    print(f"[{args.name}] SSv2 Action Recognition Accuracy: {acc*100:.2f}%")
    print(f"========================================")

    results = {
        "name": args.name,
        "dataset": "Something-Something V2",
        "task": f"frozen-encoder + attentive probe ({num_classes}-class)",
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
