"""
Temporal Attention Probe (TAP) for Diving-48 Evaluation
Keeps frozen V-JEPA2 encoder, adds temporal attention across frames
"""
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
    
    return encoder, cfg, num_frames, patch_size, tubelet_size


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
            
        v = vr.get_batch(indices)  # [T, H, W, C]
        v = v.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
        
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


class TemporalAttentionProbe(nn.Module):
    """
    Temporal Attention Probe (TAP)
    Applies self-attention across time dimension to model frame-to-frame dynamics
    """
    def __init__(self, embed_dim, num_classes, num_temporal_heads=8, num_temporal_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_temporal_layers = num_temporal_layers
        
        # CLS token for final classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Temporal attention layers - attend across time
        self.temporal_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_temporal_heads, batch_first=True)
            for _ in range(num_temporal_layers)
        ])
        self.temporal_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_temporal_layers)
        ])
        
        # Spatial aggregation (simple attention over spatial patches)
        self.spatial_pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        
        # Final classifier
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        """x: [B, T, D] - spatially-pooled per-frame features"""
        B, T, D = x.shape
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_temporal = torch.cat((cls_tokens, x), dim=1)  # [B, 1+T, D]
        for attn_layer, norm_layer in zip(self.temporal_attn_layers, self.temporal_norms):
            attn_out, _ = attn_layer(x_temporal, x_temporal, x_temporal)
            x_temporal = x_temporal + attn_out
            x_temporal = norm_layer(x_temporal)
        
        cls_out = x_temporal[:, 0]  # [B, D]
        return self.head(cls_out)


class SimpleTemporalProbe(nn.Module):
    """
    Simpler temporal probe using 1D convolutions over time
    More efficient than attention but still captures temporal patterns
    """
    def __init__(self, embed_dim, num_classes, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        # Temporal convolution layers
        self.temporal_conv = nn.Sequential(
            # Conv over time (treating spatial as channels)
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=4),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=4),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
        )
        
        # CLS token and classifier
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        """x: [B, T, D] - spatially-pooled per-frame features"""
        B, T, D = x.shape
        
        x = x.transpose(1, 2)  # [B, D, T]
        
        # Apply temporal convolutions
        x = self.temporal_conv(x)  # [B, D, T]
        
        # Transpose back: [B, T, D]
        x = x.transpose(1, 2)
        
        # Add CLS token and apply attention
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm(x)
        
        return self.head(x[:, 0])


class MotionWeightedTemporalProbe(nn.Module):
    """
    Motion-aware temporal router.
    Learns which temporal states matter, with explicit frame-difference signal.
    """
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        """x: [B, T, D] - spatially-pooled per-frame features"""
        motion = torch.zeros_like(x)
        motion[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()
        scores = self.score(torch.cat([x, motion], dim=-1)).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (weights * x).sum(dim=1)
        return self.head(pooled)


def extract_features(data_list, encoder, device, num_frames, crop_size, base_dir, patch_size, tubelet_size):
    """Extract features with spatial pooling per frame to save memory.
    Returns [N, T, D] instead of [N, T*H*W, D] (196x smaller)."""
    features = None
    labels = torch.zeros(len(data_list), dtype=torch.long)
    
    H_patches = 224 // patch_size
    W_patches = 224 // patch_size
    T_tokens = num_frames // tubelet_size
    N_spatial = H_patches * W_patches
    
    print(f"Feature extraction: T={T_tokens}, spatial={H_patches}x{W_patches}={N_spatial}, pooling spatially per-frame")
    
    valid_idx = 0
    with torch.no_grad():
        for i, item in enumerate(tqdm(data_list)):
            vid_path = os.path.join(base_dir, 'rgb', f"{item['vid_name']}.mp4")
            v = load_video(vid_path, num_frames, crop_size)
            if v is None:
                continue
            v = v.unsqueeze(0).to(device)
            
            h = encoder([v], gram_mode=False, training_mode=False)
            z = h[-1].squeeze(0)  # [L, D] on GPU
            
            L, D = z.shape
            expected_L = T_tokens * N_spatial
            if L > expected_L:
                z = z[:expected_L]
            elif L < expected_L:
                z = F.pad(z, (0, 0, 0, expected_L - L))
            
            z = z.view(T_tokens, N_spatial, D).mean(dim=1).cpu()  # [T, D]
            
            if features is None:
                features = torch.zeros((len(data_list), T_tokens, D), dtype=torch.float32)
                
            features[valid_idx] = z.float()
            labels[valid_idx] = item['label']
            valid_idx += 1
            
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
    parser.add_argument('--probe_type', type=str, default='temporal_attn', 
                        choices=['temporal_attn', 'simple_temporal', 'motion_router', 'baseline'])
    parser.add_argument('--temporal_heads', type=int, default=8)
    parser.add_argument('--temporal_layers', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    encoder, cfg, num_frames, patch_size, tubelet_size = load_model(
        args.config, args.checkpoint, device
    )
    crop_size = cfg['data']['crop_size']
    
    base_dir = '/a/mm/VJEPA2/data/diving48'
    print(f"Loading Diving-48 data from {base_dir}...")
    
    with open(os.path.join(base_dir, 'Diving48_V2_train.json'), 'r') as f:
        train_data = json.load(f)
    with open(os.path.join(base_dir, 'Diving48_V2_test.json'), 'r') as f:
        test_data = json.load(f)
        
    print(f"Train samples: {len(train_data)}, Val samples: {len(test_data)}")
    
    print("Extracting features with temporal structure preserved...")
    X_train, y_train = extract_features(
        train_data, encoder, device, num_frames, crop_size, base_dir,
        patch_size, tubelet_size
    )
    
    X_test, y_test = extract_features(
        test_data, encoder, device, num_frames, crop_size, base_dir,
        patch_size, tubelet_size
    )
    
    print(f"Train Features: {X_train.shape}, Test Features: {X_test.shape}")
    
    # Free the encoder from GPU memory
    del encoder
    torch.cuda.empty_cache()
    
    # Create probe based on type
    embed_dim = X_train.shape[-1]
    print(f"Using {args.probe_type} probe with embed_dim={embed_dim}")
    
    if args.probe_type == 'temporal_attn':
        probe = TemporalAttentionProbe(
            embed_dim=embed_dim, 
            num_classes=48,
            num_temporal_heads=args.temporal_heads,
            num_temporal_layers=args.temporal_layers
        ).to(device)
    elif args.probe_type == 'simple_temporal':
        T_tokens = 8  # num_frames // tubelet_size
        probe = SimpleTemporalProbe(
            embed_dim=embed_dim,
            num_classes=48,
            num_frames=T_tokens
        ).to(device)
    elif args.probe_type == 'motion_router':
        probe = MotionWeightedTemporalProbe(
            embed_dim=embed_dim,
            num_classes=48,
        ).to(device)
    else:
        # Fallback to standard attentive probe
        from eval_diving48 import AttentiveProbe
        probe = AttentiveProbe(embed_dim=embed_dim, num_classes=48).to(device)
    
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    batch_size = 16
    
    print(f"Training {args.probe_type} probe for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        probe.train()
        permutation = torch.randperm(X_train.size()[0])
        total_loss = 0
        num_batches = 0
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices].to(device), y_train[indices].to(device)
            
            optimizer.zero_grad()
            outputs = probe(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
            
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
    # Evaluation
    probe.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, X_test.size()[0], batch_size):
            batch_x = X_test[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size].to(device)
            outputs = probe(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).float().sum().item()
            total += batch_y.size(0)
            
    acc = correct / total
    
    print(f"\n========================================")
    print(f"[{args.name}] Diving-48 Accuracy: {acc*100:.2f}%")
    print(f"Probe: {args.probe_type}")
    print(f"========================================")

    results = {
        "name": args.name,
        "probe_type": args.probe_type,
        "dataset": "Diving-48 (V2)",
        "task": "frozen-encoder + temporal attention probe (48-class)",
        "train_probe_samples": len(X_train),
        "val_samples": len(X_test),
        "top1_accuracy_percent": round(acc * 100, 2),
        "params": {
            "lr": args.lr, 
            "weight_decay": args.wd, 
            "epochs": args.epochs,
            "temporal_heads": args.temporal_heads,
            "temporal_layers": args.temporal_layers
        },
        "checkpoint": args.checkpoint,
        "config": args.config
    }
    
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.out_json}")


if __name__ == '__main__':
    main()
