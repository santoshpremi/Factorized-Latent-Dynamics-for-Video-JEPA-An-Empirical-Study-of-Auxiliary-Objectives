import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import multiprocessing as mp
import argparse
import csv

# Constants
NUM_CLASSES = 8
FRAMES_PER_VIDEO = 16
IMG_SIZE = 224

def generate_background(bg_type, frame_idx):
    bg = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    if bg_type == 'solid':
        color = np.random.randint(0, 255, 3).tolist()
        bg[:] = color
    elif bg_type == 'gradient':
        # A simple moving gradient
        for y in range(IMG_SIZE):
            c = int((y + frame_idx * 5) % 255)
            bg[y, :] = [c, 255 - c, 128]
    elif bg_type == 'noise':
        bg = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return bg

def draw_object(frame, obj_type, x, y, size, color):
    x, y = int(x), int(y)
    size = int(size)
    if obj_type == 'circle':
        cv2.circle(frame, (x, y), size, color, -1)
    elif obj_type == 'square':
        cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), color, -1)
    elif obj_type == 'triangle':
        pts = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]], np.int32)
        cv2.fillPoly(frame, [pts], color)

def generate_video(args):
    video_path, class_id, seed = args
    np.random.seed(seed)
    random.seed(seed)
    
    frames = []
    
    # Randomize distractors
    bg_type = random.choice(['solid', 'gradient', 'noise'])
    obj_type = random.choice(['circle', 'square', 'triangle'])
    base_color = np.random.randint(50, 255, 3).tolist()
    color_shift = random.choice([True, False])
    scale_variance = random.choice([True, False])
    
    # Initial state
    x = random.uniform(IMG_SIZE * 0.2, IMG_SIZE * 0.8)
    y = random.uniform(IMG_SIZE * 0.2, IMG_SIZE * 0.8)
    vx = random.uniform(-10, 10)
    vy = random.uniform(-10, 10)
    base_size = random.uniform(10, 30)
    
    ax, ay = 0, 0
    center_x, center_y = 0, 0
    radius, angle, angular_vel = 0, 0, 0
    amplitude, phase, freq = 0, 0, 0
    x2, y2, vx2, vy2 = 0, 0, 0, 0
    obj_type2 = 'circle'
    color2 = [0, 0, 0]

    # For specific classes
    if class_id == 1: # Acceleration
        ax = random.uniform(-2, 2)
        ay = random.uniform(-2, 2)
    elif class_id == 2: # Parabolic
        vy = random.uniform(-15, -5) # Jump up
        ay = random.uniform(1, 3) # Gravity
        ax = 0
    elif class_id == 4: # Orbital
        center_x, center_y = IMG_SIZE/2, IMG_SIZE/2
        radius = random.uniform(30, 80)
        angle = random.uniform(0, 2*np.pi)
        angular_vel = random.uniform(0.1, 0.5)
    elif class_id == 5: # Harmonic
        center_x, center_y = x, y
        amplitude = random.uniform(20, 60)
        phase = random.uniform(0, 2*np.pi)
        freq = random.uniform(0.2, 0.8)
    elif class_id == 7: # Multi-body
        x2 = random.uniform(IMG_SIZE * 0.2, IMG_SIZE * 0.8)
        y2 = random.uniform(IMG_SIZE * 0.2, IMG_SIZE * 0.8)
        vx2 = random.uniform(-10, 10)
        vy2 = random.uniform(-10, 10)
        obj_type2 = random.choice(['circle', 'square', 'triangle'])
        color2 = np.random.randint(50, 255, 3).tolist()

    for i in range(FRAMES_PER_VIDEO):
        frame = generate_background(bg_type, i)
        
        # Update physics
        if class_id == 0: # Linear
            x += vx
            y += vy
        elif class_id == 1: # Acceleration
            vx += ax
            vy += ay
            x += vx
            y += vy
        elif class_id == 2: # Parabolic
            vy += ay
            x += vx
            y += vy
        elif class_id == 3: # Bounce
            x += vx
            y += vy
            if x < base_size or x > IMG_SIZE - base_size:
                vx *= -1
            if y < base_size or y > IMG_SIZE - base_size:
                vy *= -1
        elif class_id == 4: # Orbital
            angle += angular_vel
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
        elif class_id == 5: # Harmonic
            x = center_x + amplitude * np.sin(freq * i + phase)
        elif class_id == 6: # Brownian
            x += random.uniform(-15, 15)
            y += random.uniform(-15, 15)
        elif class_id == 7: # Multi-body
            # Simple attraction
            dx = x2 - x
            dy = y2 - y
            dist = max(np.sqrt(dx**2 + dy**2), 1)
            force = 50 / (dist**2)
            vx += force * (dx/dist)
            vy += force * (dy/dist)
            vx2 -= force * (dx/dist)
            vy2 -= force * (dy/dist)
            x += vx
            y += vy
            x2 += vx2
            y2 += vy2
            
        # Keep in bounds for non-bounce classes
        if class_id != 3:
            x = np.clip(x, base_size, IMG_SIZE - base_size)
            y = np.clip(y, base_size, IMG_SIZE - base_size)
            if class_id == 7:
                x2 = np.clip(x2, base_size, IMG_SIZE - base_size)
                y2 = np.clip(y2, base_size, IMG_SIZE - base_size)

        # Distractors
        current_size = base_size
        if scale_variance:
            current_size = base_size * (1 + 0.5 * np.sin(i * 0.5))
            
        current_color = base_color
        if color_shift:
            current_color = [(c + i * 10) % 255 for c in base_color]

        draw_object(frame, obj_type, x, y, current_size, current_color)
        if class_id == 7:
            draw_object(frame, obj_type2, x2, y2, current_size, color2)
            
        frames.append(frame)

    # Save video
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(video_path, frames, fps=10, format='FFMPEG', codec='libx264', macro_block_size=None)
    except Exception as e:
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (IMG_SIZE, IMG_SIZE))
        for frame in frames:
            out.write(frame)
        out.release()
    return video_path, class_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/a/mm/VJEPA2/data/KathaBench-V1')
    parser.add_argument('--num_videos', type=int, default=50000)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    splits = {
        'train': int(args.num_videos * 0.7),
        'val': int(args.num_videos * 0.1),
        'test': args.num_videos - int(args.num_videos * 0.7) - int(args.num_videos * 0.1)
    }
    
    tasks = []
    for split, count in splits.items():
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for i in range(count):
            class_id = i % NUM_CLASSES
            video_path = os.path.join(split_dir, f"{class_id}_{i}.mp4")
            tasks.append((video_path, class_id, random.randint(0, 1000000)))

    print(f"Generating {args.num_videos} videos across {args.num_workers} workers...")
    
    csv_path = os.path.join(args.output_dir, 'labels.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'label', 'split'])
        
        with mp.Pool(args.num_workers) as pool:
            for video_path, class_id in tqdm(pool.imap_unordered(generate_video, tasks), total=len(tasks)):
                split = video_path.split('/')[-2]
                writer.writerow([video_path, class_id, split])

if __name__ == '__main__':
    main()
