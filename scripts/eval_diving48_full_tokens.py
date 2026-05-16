"""
Full-token Diving-48 probes for frozen V-JEPA2 features.

Unlike eval_diving48_tap.py, this preserves spatial patch tokens as [B, T, N, D].
This directly tests whether Diving-48 needs pose/body-part information that was
destroyed by spatial pooling.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../vjepa2"))
from app.vjepa_2_1.utils import init_video_model


def load_model(config_path, checkpoint_path, device):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    encoder, _ = init_video_model(
        device=device,
        patch_size=cfg["data"]["patch_size"],
        max_num_frames=cfg["data"]["dataset_fpcs"][0],
        tubelet_size=cfg["data"]["tubelet_size"],
        model_name=cfg["model"]["model_name"],
        crop_size=cfg["data"]["crop_size"],
        pred_depth=8,
        use_rope=cfg["model"].get("use_rope", False),
        modality_embedding=cfg["model"].get("modality_embedding", False),
        interpolate_rope=cfg["model"].get("interpolate_rope", False),
        use_sdpa=cfg["meta"].get("use_sdpa", False),
        uniform_power=cfg["model"].get("uniform_power", False),
        use_mask_tokens=cfg["model"].get("use_mask_tokens", False),
        zero_init_mask_tokens=cfg["model"].get("zero_init_mask_tokens", False),
        has_cls_first=cfg["model"].get("has_cls_first", False),
        img_temporal_dim_size=cfg["model"].get("img_temporal_dim_size", None),
        n_registers=cfg["model"].get("n_registers", 0),
        n_registers_predictor=cfg["model"].get("n_registers_predictor", 0),
    )

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["target_encoder"].items()}
    encoder.load_state_dict(state_dict, strict=False)
    encoder.to(device)
    encoder.eval()
    return encoder, cfg


def load_video(path, num_frames, crop_size):
    import decord

    decord.bridge.set_bridge("torch")
    try:
        vr = decord.VideoReader(path, num_threads=1)
        total_frames = len(vr)
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = np.pad(np.arange(total_frames), (0, num_frames - total_frames), mode="edge")

        v = vr.get_batch(indices).permute(3, 0, 1, 2).float() / 255.0
        _, _, height, width = v.shape
        scale = crop_size / min(height, width)
        new_height, new_width = int(height * scale), int(width * scale)
        v = F.interpolate(v, size=(new_height, new_width), mode="bilinear", align_corners=False)

        top = (new_height - crop_size) // 2
        left = (new_width - crop_size) // 2
        v = v[:, :, top : top + crop_size, left : left + crop_size]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        return (v - mean) / std
    except Exception as exc:
        print(f"Failed to load {path}: {exc}")
        return None


def extract_full_tokens(data_list, encoder, device, cfg, base_dir):
    num_frames = cfg["data"]["dataset_fpcs"][0]
    crop_size = cfg["data"]["crop_size"]
    patch_size = cfg["data"]["patch_size"]
    tubelet_size = cfg["data"]["tubelet_size"]
    t_tokens = num_frames // tubelet_size
    h_patches = crop_size // patch_size
    w_patches = crop_size // patch_size
    n_spatial = h_patches * w_patches
    expected_tokens = t_tokens * n_spatial

    print(f"Extracting full tokens: T={t_tokens}, spatial={h_patches}x{w_patches}={n_spatial}")
    labels = torch.zeros(len(data_list), dtype=torch.long)
    features = None
    valid_idx = 0

    with torch.no_grad():
        for item in tqdm(data_list):
            video_path = os.path.join(base_dir, "rgb", f"{item['vid_name']}.mp4")
            video = load_video(video_path, num_frames, crop_size)
            if video is None:
                continue

            video = video.unsqueeze(0).to(device)
            h = encoder([video], gram_mode=False, training_mode=False)
            z = h[-1].squeeze(0)
            length, dim = z.shape

            if length > expected_tokens:
                z = z[:expected_tokens]
            elif length < expected_tokens:
                z = F.pad(z, (0, 0, 0, expected_tokens - length))

            z = z.view(t_tokens, n_spatial, dim).cpu().half()
            if features is None:
                features = torch.empty((len(data_list), t_tokens, n_spatial, dim), dtype=torch.float16)

            features[valid_idx] = z
            labels[valid_idx] = item["label"]
            valid_idx += 1

    return features[:valid_idx], labels[:valid_idx]


class SpatioTemporalAttentionProbe(nn.Module):
    def __init__(self, embed_dim, num_classes, temporal_heads=8, spatial_heads=8):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(embed_dim, temporal_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.spatial_cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, spatial_heads, batch_first=True)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        # x: [B, T, N, D]
        batch, steps, patches, dim = x.shape
        tracks = x.permute(0, 2, 1, 3).reshape(batch * patches, steps, dim)
        temporal_out, _ = self.temporal_attn(tracks, tracks, tracks)
        tracks = self.temporal_norm(tracks + temporal_out)
        patch_features = tracks.mean(dim=1).view(batch, patches, dim)

        cls = self.spatial_cls.expand(batch, -1, -1)
        tokens = torch.cat([cls, patch_features], dim=1)
        spatial_out, _ = self.spatial_attn(tokens, tokens, tokens)
        tokens = self.spatial_norm(tokens + spatial_out)
        return self.head(tokens[:, 0])


class SpatialRouterProbe(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
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
        batch, steps, patches, dim = x.shape
        tokens = x.reshape(batch, steps * patches, dim)
        scores = self.score(tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (weights * tokens).sum(dim=1)
        return self.head(pooled)


class MotionTokenRouterProbe(nn.Module):
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
        # x: [B, T, N, D]
        motion = torch.zeros_like(x)
        motion[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()
        tokens = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        motion_tokens = motion.reshape_as(tokens)
        scores = self.score(torch.cat([tokens, motion_tokens], dim=-1)).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (weights * tokens).sum(dim=1)
        return self.head(pooled)


def make_probe(probe_type, embed_dim, num_classes, temporal_heads):
    if probe_type == "st_attention":
        return SpatioTemporalAttentionProbe(embed_dim, num_classes, temporal_heads=temporal_heads)
    if probe_type == "spatial_router":
        return SpatialRouterProbe(embed_dim, num_classes)
    if probe_type == "motion_router_full":
        return MotionTokenRouterProbe(embed_dim, num_classes)
    raise ValueError(f"Unknown probe_type: {probe_type}")


def evaluate(probe, x_test, y_test, device, batch_size):
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, x_test.size(0), batch_size):
            batch_x = x_test[start : start + batch_size].to(device).float()
            batch_y = y_test[start : start + batch_size].to(device)
            logits = probe(batch_x)
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).float().sum().item()
            total += batch_y.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--probe_type", required=True, choices=["st_attention", "spatial_router", "motion_router_full"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temporal_heads", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoder, cfg = load_model(args.config, args.checkpoint, device)

    base_dir = "/a/mm/VJEPA2/data/diving48"
    with open(os.path.join(base_dir, "Diving48_V2_train.json"), "r") as f:
        train_data = json.load(f)
    with open(os.path.join(base_dir, "Diving48_V2_test.json"), "r") as f:
        test_data = json.load(f)

    x_train, y_train = extract_full_tokens(train_data, encoder, device, cfg, base_dir)
    x_test, y_test = extract_full_tokens(test_data, encoder, device, cfg, base_dir)
    print(f"Train Features: {x_train.shape}, Test Features: {x_test.shape}")

    del encoder
    torch.cuda.empty_cache()

    embed_dim = x_train.shape[-1]
    probe = make_probe(args.probe_type, embed_dim, 48, args.temporal_heads).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training {args.name} for {args.epochs} epochs, batch_size={args.batch_size}")
    for epoch in range(args.epochs):
        probe.train()
        permutation = torch.randperm(x_train.size(0))
        total_loss = 0.0
        batches = 0
        for start in range(0, x_train.size(0), args.batch_size):
            idx = permutation[start : start + args.batch_size]
            batch_x = x_train[idx].to(device).float()
            batch_y = y_train[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            acc = evaluate(probe, x_test, y_test, device, args.batch_size)
            print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss / batches:.4f} val_acc={acc*100:.2f}%")

    acc = evaluate(probe, x_test, y_test, device, args.batch_size)
    print(f"\n[{args.name}] Diving-48 Accuracy: {acc*100:.2f}%")

    results = {
        "name": args.name,
        "probe_type": args.probe_type,
        "dataset": "Diving-48 (V2)",
        "task": "frozen-encoder + full-spatiotemporal-token probe (48-class)",
        "train_probe_samples": len(x_train),
        "val_samples": len(x_test),
        "top1_accuracy_percent": round(acc * 100, 2),
        "params": {
            "lr": args.lr,
            "weight_decay": args.wd,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "temporal_heads": args.temporal_heads,
        },
        "checkpoint": args.checkpoint,
        "config": args.config,
    }
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.out_json}")


if __name__ == "__main__":
    main()
