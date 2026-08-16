import os
import glob
import numpy as np
import torch
import csv
import utils

OUT_DIR = 'outputs/restored_eval'
GT_DIR = 'datasets/train/train/GT'

paths = sorted(glob.glob(os.path.join(OUT_DIR, '*.npy')))
if not paths:
    print('No outputs found in', OUT_DIR)
    raise SystemExit(1)

rows = []
psnrs = []
ssims = []
for p in paths:
    fname = os.path.basename(p)
    gt_path = os.path.join(GT_DIR, fname)
    if not os.path.exists(gt_path):
        print('GT missing for', fname, '- skipping')
        continue
    out = np.load(p)
    gt = np.load(gt_path).astype(np.float32)
    # normalize if needed
    if out.dtype != np.float32:
        out = out.astype(np.float32)
    if gt.dtype != np.float32:
        gt = gt.astype(np.float32)
    # Ensure channel-first (1,H,W)
    if out.ndim == 2:
        out = out[np.newaxis, ...]
    if gt.ndim == 2:
        gt = gt[np.newaxis, ...]
    # Batch dim
    out_t = torch.from_numpy(out).unsqueeze(0)
    gt_t = torch.from_numpy(gt).unsqueeze(0)
    # Calculate
    try:
        psnr = utils.calculate_psnr(out_t, gt_t).item()
    except Exception:
        psnr = float(utils.calculate_psnr(out_t, gt_t))
    ssim = float(utils.calculate_ssim(out_t, gt_t))
    rows.append((fname, psnr, ssim))
    psnrs.append(psnr)
    ssims.append(ssim)
    if len(rows) % 50 == 0:
        print('Processed', len(rows))

out_csv = os.path.join(OUT_DIR, 'metrics.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['filename','psnr_db','ssim'])
    w.writerows(rows)

print('Wrote', out_csv)
print('Count', len(rows))
print('Mean PSNR', sum(psnrs)/len(psnrs) if psnrs else 0)
print('Mean SSIM', sum(ssims)/len(ssims) if ssims else 0)
