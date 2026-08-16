import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SemiconductorDataset(Dataset):
    def __init__(self, degraded_dir, gt_dir, augment=True):
        self.degraded_dir = degraded_dir
        self.gt_dir = gt_dir
        self.augment = augment

        # List all .npy files in degraded folder
        self.degraded_files = sorted([
            f for f in os.listdir(degraded_dir) 
            if f.endswith('.npy')
        ])

        if len(self.degraded_files) == 0:
            raise ValueError(f"No .npy files found in: {degraded_dir}")

        # Verify matching GT files exist
        self.valid_pairs = []
        for f in self.degraded_files:
            gt_path = os.path.join(gt_dir, f)
            if os.path.exists(gt_path):
                self.valid_pairs.append(f)
            else:
                print(f"Warning: GT file missing for {f}, skipping.")

        if len(self.valid_pairs) == 0:
            raise ValueError(f"No matching GT files found in: {gt_dir}")

        print(f"Dataset loaded: {len(self.valid_pairs)} paired samples.")

    def __len__(self):
        return len(self.valid_pairs)

    def _load_npy(self, path):
        """Load .npy and normalize to (1, H, W) float32."""
        arr = np.load(path).astype(np.float32)

        # Handle various shapes: (H,W), (1,H,W), (H,W,1)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]           # (1, H, W)
        elif arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = arr[..., 0]                 # (H, W)
                arr = arr[np.newaxis, ...]        # (1, H, W)
            elif arr.shape[0] != 1:
                # Assume first dim is channel or squeeze
                arr = arr[np.newaxis, ...] if arr.shape[0] > 1 else arr

        return arr

    def __getitem__(self, idx):
        fname = self.valid_pairs[idx]

        deg_path = os.path.join(self.degraded_dir, fname)
        gt_path = os.path.join(self.gt_dir, fname)

        degraded = self._load_npy(deg_path)   # (1, H, W)
        gt = self._load_npy(gt_path)          # (1, 2H, 2W)

        degraded = torch.from_numpy(degraded).float()
        gt = torch.from_numpy(gt).float()

        # CRITICAL: Do NOT clamp degraded input (speckle pushes values beyond [0,1])
        # GT is always [0,1] per problem statement
        gt = torch.clamp(gt, 0.0, 1.0)

        # Simple augment: random horizontal flip
        if self.augment and torch.rand(1) > 0.5:
            degraded = torch.flip(degraded, dims=[2])
            gt = torch.flip(gt, dims=[2])

        return degraded, gt