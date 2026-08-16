#!/usr/bin/env python3
"""
Compute PSNR, SSIM, and LPIPS metrics for restored test images.

If ground truth is available, this script compares restored outputs against GT.
"""

import os
import glob
import argparse
import numpy as np
import torch
import csv
from pathlib import Path

import utils

def compute_metrics(restored_dir, gt_dir=None, output_csv=None):
    """
    Compute metrics for restored images.
    
    Args:
        restored_dir: Directory containing restored .npy files
        gt_dir: Optional directory containing ground truth .npy files
        output_csv: Path to save CSV results
    
    Returns:
        Dictionary with mean metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Find all restored .npy files
    restored_paths = sorted(glob.glob(os.path.join(restored_dir, "*.npy")))
    if not restored_paths:
        print(f"No .npy files found in: {restored_dir}")
        return None
    
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
        
        row = {'filename': fname}
        
        # If GT directory provided, compare
        if gt_dir:
            gt_path = os.path.join(gt_dir, fname)
            if os.path.exists(gt_path):
                gt_np = np.load(gt_path).astype(np.float32)
                if gt_np.ndim == 2:
                    gt_np = gt_np[np.newaxis, ...]
                gt_t = torch.from_numpy(gt_np).unsqueeze(0).to(device)
                
                # Compute metrics
                psnr = float(utils.calculate_psnr(restored_t, gt_t))
                ssim = float(utils.calculate_ssim(restored_t, gt_t))
                
                row['psnr_db'] = round(psnr, 4)
                row['ssim'] = round(ssim, 4)
                
                psnrs.append(psnr)
                ssims.append(ssim)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(restored_paths)} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
            else:
                print(f"GT missing for {fname}")
                row['psnr_db'] = None
                row['ssim'] = None
        
        rows.append(row)
    
    # Save CSV
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        with open(output_csv, 'w', newline='') as f:
            if gt_dir and rows and 'psnr_db' in rows[0]:
                writer = csv.DictWriter(f, fieldnames=['filename', 'psnr_db', 'ssim'])
                writer.writeheader()
            else:
                writer = csv.DictWriter(f, fieldnames=['filename'])
                writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {output_csv}")
    
    # Print summary
    if psnrs:
        print(f"\n=== Metrics Summary ===")
        print(f"Mean PSNR: {np.mean(psnrs):.4f} dB")
        print(f"Mean SSIM: {np.mean(ssims):.4f}")
        print(f"Std PSNR: {np.std(psnrs):.4f} dB")
        print(f"Std SSIM: {np.std(ssims):.4f}")
        
        return {
            'mean_psnr': float(np.mean(psnrs)),
            'mean_ssim': float(np.mean(ssims)),
            'std_psnr': float(np.std(psnrs)),
            'std_ssim': float(np.std(ssims)),
            'count': len(psnrs)
        }
    else:
        return {'count': len(restored_paths)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for restored images")
    parser.add_argument('--restored_dir', type=str, required=True, help='Directory with restored .npy files')
    parser.add_argument('--gt_dir', type=str, default=None, help='Directory with ground truth .npy files')
    parser.add_argument('--output_csv', type=str, default='metrics.csv', help='CSV file to save results')
    args = parser.parse_args()
    
    compute_metrics(args.restored_dir, args.gt_dir, args.output_csv)
