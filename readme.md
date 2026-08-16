# KLA Semiconductor Image Restoration Pipeline

This project restores degraded semiconductor microscopy inputs using a residual dense network based on RRDB-style reconstruction. The workflow includes:

- Training on paired degraded/ground-truth `.npy` datasets
- Evaluation on unseen degraded inputs
- Visualization of noisy/restored/ground-truth sample comparisons
- PSNR/SSIM metric computation

## Project Structure

```
.
├── README.md                          # This file - complete setup instructions
├── requirements.txt                   # Complete pip freeze for reproducibility
├── automate_pipeline.py               # Unified automation entry point
│
├── model.py                           # RRDB-based restoration network architecture
├── dataset.py                         # Dataset loader for .npy image pairs
├── train.py                           # Training script (reproduces training from scratch)
├── evaluate.py                        # Evaluation/inference script (standalone, no manual edits needed)
├── visualize.py                       # Result visualization helper
├── compute_metrics.py                 # PSNR/SSIM metric calculation
├── utils.py                           # Utility functions
├── losses.py                          # Loss functions
│
├── checkpoints/                       # Trained model weights
│   └── best_model.pth                 # Final trained model (17.4 MB) ← Use this for inference
│
├── datasets/                          # Training and test datasets
│   ├── train/
│   │   └── train/
│   │       ├── NoisyLR/               # 3200 degraded input images
│   │       └── GT/                    # 3200 ground-truth images
│   └── Test_NoisyLR/
│       └── NoisyLR/                   # 400 test degraded input images
│
└── outputs/                           # Model outputs
    └── restored_eval/                 # 401 restored test images (from best_model.pth)
```

## Requirements

Install dependencies from the complete pip freeze output (ensures reproducibility):

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run Evaluation Only (Recommended for reviewers)

The evaluation script runs **standalone without manual edits**. It accepts:
- Path to degraded test images
- Path to output directory

```bash
python evaluate.py \
  --input_dir datasets/Test_NoisyLR/NoisyLR \
  --output_dir outputs/restored_eval \
  --model_path checkpoints/best_model.pth
```

**Output:** Restored `.npy` files saved to `outputs/restored_eval/`

### Option 2: Run Full Training Pipeline

```bash
python automate_pipeline.py train \
  --train_degraded datasets/train/train/NoisyLR \
  --train_gt datasets/train/train/GT \
  --checkpoint_dir checkpoints \
  --epochs 300 \
  --batch_size 4
```

### Option 3: Run Complete Workflow (Train → Evaluate → Visualize → Metrics)

```bash
python automate_pipeline.py full \
  --train_degraded datasets/train/train/NoisyLR \
  --train_gt datasets/train/train/GT \
  --input_dir datasets/Test_NoisyLR/NoisyLR \
  --output_dir outputs/restored_eval \
  --model_path checkpoints/best_model.pth
```

## Submission Checklist

✅ **README.md** — Complete setup instructions; reviewers can clone and run inference without contacting you  
✅ **Evaluation Script** — `evaluate.py` is standalone; accepts input/output paths; runs without manual edits  
✅ **Training Script** — `train.py` reproduces training from scratch  
✅ **Trained Model Weights** — `checkpoints/best_model.pth` (17.4 MB); downloaded via Git  
✅ **Restored Test Outputs** — `outputs/restored_eval/` contains 401 restored images  
✅ **requirements.txt** — Complete pip freeze for reproducibility

## Evaluation Script Details

The evaluation script (`evaluate.py`):
- Loads your trained model from disk
- Runs inference on all `.npy` files in the input directory
- Writes restored outputs to the specified output directory
- Runs without any manual code edits or configuration

**Command:**
```bash
python evaluate.py \
  --input_dir <path_to_test_images> \
  --output_dir <path_to_save_outputs> \
  --model_path checkpoints/best_model.pth
```

## Data Format

- **Input images:** `.npy` files (single-channel grayscale, shape: (H, W) or (1, H, W))
- **Output images:** `.npy` files (single-channel, shape: (1, 2H, 2W) for 2× upscaling)
- **Value range:** Restored outputs are clamped to [0, 1]

## Model Details

- **Architecture:** RRDB-Net (Residual Dense Block Network)
- **Input:** Low-resolution degraded images
- **Output:** 2× upscaled restored images
- **Training Loss:** Charbonnier loss
- **Optimizer:** AdamW with cosine annealing scheduler
- **Training:** 300 epochs, batch size 4 (on GPU)

## Reproduction Notes

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run inference: `python evaluate.py --input_dir datasets/Test_NoisyLR/NoisyLR --output_dir outputs/restored_eval --model_path checkpoints/best_model.pth`
4. Review outputs in `outputs/restored_eval/`

All scripts are production-ready and fully automated.

## Citation / Use

This repository is intended for semiconductor image restoration tasks and can be adapted for custom degraded datasets with the same file naming and folder conventions.

```

### 3) Visualize a few samples

```bash
python automate_pipeline.py visualize \
  --model_path checkpoints/best_model.pth \
  --degraded_path datasets/train/train/NoisyLR \
  --gt_path datasets/train/train/GT \
  --num_samples 3 \
  --out outputs/visualization.png
```

### 4) Compute metrics

```bash
python automate_pipeline.py metrics
```

### Full automated pipeline

```bash
python automate_pipeline.py full \
  --train_degraded datasets/train/train/NoisyLR \
  --train_gt datasets/train/train/GT \
  --input_dir datasets/Test_NoisyLR/NoisyLR \
  --output_dir outputs/restored_eval \
  --model_path checkpoints/best_model.pth
```

## Notes

- Training expects GPU access for practical speed.
- The dataset uses `.npy` files and expects matching file names between degraded input and ground truth folders.
- Outputs are saved under `outputs/restored_eval` and metrics are written to the same directory as `metrics.csv`.

## Citation / use

This repository is intended for semiconductor image restoration tasks and can be adapted for custom degraded datasets with the same file naming and folder conventions.
