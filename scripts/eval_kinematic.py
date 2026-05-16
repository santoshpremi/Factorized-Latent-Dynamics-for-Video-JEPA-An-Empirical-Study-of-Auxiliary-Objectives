import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import yaml

# Add vjepa2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../vjepa2'))
from app.vjepa_2_1.utils import init_video_model
from src.datasets.data_manager import init_data

def load_model(config_path, checkpoint_path, device):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model_name = cfg['model']['model_name']
    patch_size = cfg['data']['patch_size']
    crop_size = cfg['data']['crop_size']
    tubelet_size = cfg['data']['tubelet_size']
    
    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=8,
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
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract encoder weights from target_encoder (or encoder)
    encoder_state_dict = {}
    for k, v in checkpoint['target_encoder'].items():
        # Remove 'module.' prefix from DDP
        new_k = k.replace('module.', '')
        encoder_state_dict[new_k] = v
        
    encoder.load_state_dict(encoder_state_dict, strict=False)
    encoder.to(device)
    encoder.eval()
    
    return encoder, cfg

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config_path = '/a/mm/VJEPA2/configs/train_synth_2gpu_kinematic.yaml'
    checkpoint_path = '/a/mm/VJEPA2/runs/train_synth_2gpu_kinematic/latest.pth.tar'
    
    print("Loading model...")
    encoder, cfg = load_model(config_path, checkpoint_path, device)
    
    print("Loading data...")
    
    import pandas as pd
    import torchvision.io as tvio
    import imageio.v2 as imageio
    
    def load_video(path, num_frames=8):
        try:
            v, _, _ = tvio.read_video(path, pts_unit='sec')
            v = v[:num_frames]
        except Exception as e:
            try:
                reader = imageio.get_reader(path)
                v = []
                for i, frame in enumerate(reader):
                    if i >= num_frames: break
                    v.append(frame)
                v = torch.from_numpy(np.stack(v))
            except Exception as e2:
                print(f"Failed to load {path}: {e2}")
                return None
            
        # v is [T, H, W, C] -> [C, T, H, W]
        v = v.permute(3, 0, 1, 2).float() / 255.0
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        v = (v - mean) / std
        return v

    df = pd.read_csv('/a/mm/VJEPA2/data/synth/synth_eval.csv', sep=' ', header=None, names=['path', 'label'])
    
    print("Extracting features...")
    features = []
    labels = []
    
    with torch.no_grad():
        for i, row in df.iterrows():
            v = load_video(row['path'])
            if v is None:
                continue
            v = v.unsqueeze(0).to(device) # [1, C, T, H, W]
            # Forward pass
            h = encoder([v], gram_mode=False, training_mode=False)
            # h is a list of tensors, we take the last layer output
            z = h[-1] # [1, L, D]
            
            features.append(z.squeeze(0).cpu()) # [L, D]
            labels.append(row['label'])
            
            if (i+1) % 50 == 0:
                print(f"  Processed {i+1}/{len(df)}")
                
    X = torch.stack(features) # [N, L, D]
    y = torch.tensor(labels) # [N]
    
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    
    # Train/Test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print("Training Attentive Probe...")
    class AttentiveProbe(nn.Module):
        def __init__(self, embed_dim, num_classes, num_heads=8):
            super().__init__()
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)
            
        def forward(self, x):
            B = x.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x_attn, _ = self.attn(x, x, x)
            x = x + x_attn
            x = self.norm(x)
            cls_out = x[:, 0]
            return self.head(cls_out)

    probe = AttentiveProbe(embed_dim=X.shape[-1], num_classes=8).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    batch_size = 32
    epochs = 100
    
    for epoch in range(epochs):
        probe.train()
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = probe(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    probe.eval()
    with torch.no_grad():
        outputs = probe(X_test)
        _, preds = torch.max(outputs, 1)
        acc = (preds == y_test).float().mean().item()
    
    print(f"\n========================================")
    print(f"Kinematic-JEPA Motion Probe Accuracy: {acc*100:.2f}%")
    print(f"========================================")
    print("Random chance would be 12.5% (8 directions)")
    
if __name__ == '__main__':
    main()

