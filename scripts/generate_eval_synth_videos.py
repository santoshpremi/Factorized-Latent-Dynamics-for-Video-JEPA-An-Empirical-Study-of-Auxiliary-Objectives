import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

def _write_video(path: str, frames_thwc_uint8: torch.Tensor, fps: int) -> None:
    try:
        import torchvision.io as tvio
        tvio.write_video(
            path,
            frames_thwc_uint8,
            fps=fps,
            video_codec="h264",
            options={"crf": "23", "pix_fmt": "yuv420p"},
        )
        return
    except Exception as e:
        pass

    import imageio.v2 as imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    try:
        for frame in frames_thwc_uint8.numpy():
            writer.append_data(frame)
    finally:
        writer.close()

def make_clip(num_frames: int, h: int, w: int, seed: int, dx: int, dy: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)

    bg_color = rng.integers(0, 80, size=3, dtype=np.uint8)
    frames[:] = bg_color[None, None, None, :]
    frames += (rng.integers(0, 30, size=(num_frames, h, w, 1)).astype(np.uint8))

    blob_color = rng.integers(120, 256, size=3, dtype=np.uint8)
    blob_r = rng.integers(10, min(h, w) // 4)
    x0, y0 = rng.integers(blob_r, w - blob_r), rng.integers(blob_r, h - blob_r)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for t in range(num_frames):
        cx = int(np.clip(x0 + dx * t, blob_r, w - blob_r - 1))
        cy = int(np.clip(y0 + dy * t, blob_r, h - blob_r - 1))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= blob_r ** 2
        frames[t][mask] = blob_color
    return torch.from_numpy(frames)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--num-clips", type=int, default=200)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Define 8 directions (classes 0-7)
    directions = [
        (0, -4),  # N (0)
        (3, -3),  # NE (1)
        (4, 0),   # E (2)
        (3, 3),   # SE (3)
        (0, 4),   # S (4)
        (-3, 3),  # SW (5)
        (-4, 0),  # W (6)
        (-3, -3), # NW (7)
    ]

    rows = []
    for i in range(args.num_clips):
        label = i % 8
        dx, dy = directions[label]
        clip = make_clip(args.num_frames, args.height, args.width, seed=args.seed + i, dx=dx, dy=dy)
        clip_path = out_dir / f"eval_synth_{i:04d}.mp4"
        _write_video(str(clip_path), clip, fps=args.fps)
        rows.append(f"{clip_path.resolve()} {label}")
        if (i + 1) % 10 == 0 or i == args.num_clips - 1:
            print(f"  wrote {i + 1}/{args.num_clips}")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(rows) + "\n")
    print(f"Wrote {args.num_clips} clips to {out_dir}")
    print(f"Wrote CSV to {csv_path}")

if __name__ == "__main__":
    main()
