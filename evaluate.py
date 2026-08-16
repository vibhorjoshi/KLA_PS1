#!/usr/bin/env python3
"""
Standalone Evaluation for KLA Hackathon.
Reads .npy files from input_dir, saves restored .npy files to output_dir.
"""
import os
import argparse
import time
import glob
import numpy as np
import torch

from model import RestorationNet

def load_npy(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
        arr = arr[np.newaxis, ...]
    return arr

def save_npy(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

@torch.inference_mode()
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = RestorationNet(in_channels=1, out_channels=1, features=64, num_rrdb=6, upscale=2).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model: {args.model_path}")

    # Find all .npy files
    input_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if len(input_paths) == 0:
        raise ValueError(f"No .npy files found in: {args.input_dir}")

    print(f"Found {len(input_paths)} test images.")
    os.makedirs(args.output_dir, exist_ok=True)

    # Warmup
    dummy = torch.randn(1, 1, 128, 128).to(device)
    _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    total_time = 0.0

    for img_path in input_paths:
        fname = os.path.basename(img_path)
        name = fname.replace('.npy', '')
        out_path = os.path.join(args.output_dir, f"{name}.npy")

        img_np = load_npy(img_path)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

        start = time.time()
        output = model(img_tensor)
        output = torch.clamp(output, 0.0, 1.0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        total_time += elapsed

        out_np = output.squeeze(0).cpu().numpy()
        save_npy(out_np, out_path)

        print(f"Processed {fname} | Time: {elapsed*1000:.2f} ms")

    avg = total_time / len(input_paths)
    print(f"\nTotal: {total_time:.4f}s | Avg: {avg*1000:.2f} ms/image | Saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Folder with degraded .npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder to save restored .npy files')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth', help='Path to .pth weights')
    args = parser.parse_args()
    main(args)