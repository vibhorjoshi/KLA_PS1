#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import torch
import csv
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. PNG generation disabled. Install with: pip install Pillow")

import utils

def normalize_for_display(img_np):
    """Normalize image to [0, 255] for PNG display."""
    if img_np.ndim == 3 and img_np.shape[0] == 1:
        img_np = img_np[0]
    
    # Clip to valid range
    img_np = np.clip(img_np, 0, 1)
    
    # Convert to 8-bit
    img_uint8 = (img_np * 255).astype(np.uint8)
    return img_uint8

def create_comparison_png(degraded_np, restored_np, gt_np, output_path):
    """Create side-by-side PNG comparison."""
    if not PIL_AVAILABLE:
        return
    
    # Normalize
    deg_img = normalize_for_display(degraded_np)
    res_img = normalize_for_display(restored_np)
    gt_img = normalize_for_display(gt_np)
    
    # Ensure same height
    h = min(deg_img.shape[0], res_img.shape[0], gt_img.shape[0])
    w = min(deg_img.shape[1], res_img.shape[1], gt_img.shape[1])
    
    deg_img = deg_img[:h, :w]
    res_img = res_img[:h, :w]
    gt_img = gt_img[:h, :w]
    
    # Create PIL images
    deg_pil = Image.fromarray(deg_img, mode='L')
    res_pil = Image.fromarray(res_img, mode='L')
    gt_pil = Image.fromarray(gt_img, mode='L')
    
    # Create side-by-side
    total_width = w * 3 + 20  # Add spacing
    total_height = h
    
    comparison = Image.new('L', (total_width, total_height), 255)
    comparison.paste(deg_pil, (0, 0))
    comparison.paste(res_pil, (w + 10, 0))
    comparison.paste(gt_pil, (w * 2 + 20, 0))
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison.save(output_path)

def generate_comparisons(degraded_dir, restored_dir, gt_dir, output_dir):
    """
    Generate comparison PNGs and metrics CSV.
    
    Args:
        degraded_dir: Directory with degraded (noisy) .npy files
        restored_dir: Directory with restored .npy files
        gt_dir: Directory with ground truth .npy files
        output_dir: Output directory for PNGs and CSV
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    comparisons_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Find all restored .npy files
    restored_paths = sorted(glob.glob(os.path.join(restored_dir, "*.npy")))
    print(f"Found {len(restored_paths)} restored images")
    
    rows = []
    psnrs = []
    ssims = []
    
    for i, restored_path in enumerate(restored_paths):
        fname = os.path.basename(restored_path)
        name = fname.replace('.npy', '')
        
        # Load restored
        restored_np = np.load(restored_path).astype(np.float32)
        if restored_np.ndim == 2:
            restored_np = restored_np[np.newaxis, ...]
        restored_t = torch.from_numpy(restored_np).unsqueeze(0).to(device)
        
        # Check for degraded and GT
        deg_path = os.path.join(degraded_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        
        row = {'filename': fname}
        
        if os.path.exists(deg_path) and os.path.exists(gt_path):
            # Load degraded and GT
            deg_np = np.load(deg_path).astype(np.float32)
            if deg_np.ndim == 2:
                deg_np = deg_np[np.newaxis, ...]
            
            gt_np = np.load(gt_path).astype(np.float32)
            if gt_np.ndim == 2:
                gt_np = gt_np[np.newaxis, ...]
            
            # Compute metrics
            restored_t_metric = torch.from_numpy(restored_np).unsqueeze(0).to(device)
            gt_t = torch.from_numpy(gt_np).unsqueeze(0).to(device)
            
            psnr = float(utils.calculate_psnr(restored_t_metric, gt_t))
            ssim = float(utils.calculate_ssim(restored_t_metric, gt_t))
            
            row['psnr_db'] = round(psnr, 4)
            row['ssim'] = round(ssim, 4)
            
            psnrs.append(psnr)
            ssims.append(ssim)
            
            # Create comparison PNG
            if PIL_AVAILABLE:
                png_path = os.path.join(comparisons_dir, f"{name}.png")
                create_comparison_png(deg_np, restored_np, gt_np, png_path)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(restored_paths)} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
        else:
            if not os.path.exists(deg_path):
                print(f"Degraded missing for {fname}")
            if not os.path.exists(gt_path):
                print(f"GT missing for {fname}")
            row['psnr_db'] = None
            row['ssim'] = None
        
        rows.append(row)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['filename', 'psnr_db', 'ssim']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {csv_path}")
    
    # Print summary
    if psnrs:
        print(f"\n=== Comparison Summary ===")
        print(f"Mean PSNR: {np.mean(psnrs):.4f} dB")
        print(f"Mean SSIM: {np.mean(ssims):.4f}")
        print(f"Comparisons saved to: {comparisons_dir}")
    
    return {
        'mean_psnr': float(np.mean(psnrs)) if psnrs else None,
        'mean_ssim': float(np.mean(ssims)) if ssims else None,
        'count': len(psnrs)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparison PNGs and metrics CSV")
    parser.add_argument('--degraded_dir', type=str, required=True, help='Directory with degraded .npy files')
    parser.add_argument('--restored_dir', type=str, required=True, help='Directory with restored .npy files')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory with ground truth .npy files')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for PNGs and CSV')
    args = parser.parse_args()
    
    generate_comparisons(args.degraded_dir, args.restored_dir, args.gt_dir, args.output_dir)
