# KLA Semiconductor Image Restoration Pipeline

This project restores degraded semiconductor microscopy inputs using a residual dense network based on RRDB-style reconstruction. The workflow includes:

- Training on paired degraded/ground-truth `.npy` datasets
- Evaluation on unseen degraded inputs
- Visualization of noisy/restored/ground-truth sample comparisons
- PSNR/SSIM metric computation

## Project structure

- `model.py` — restoration network architecture
- `dataset.py` — dataset loader for `.npy` image pairs
- `train.py` — training script
- `evaluate.py` — inference script
- `visualize.py` — result visualization script
- `compute_metrics.py` — metric calculation for generated outputs
- `automate_pipeline.py` — single entry point for the full workflow
- `checkpoints/` — saved model checkpoints
- `datasets/` — training/test dataset folders
- `outputs/` — restored evaluation outputs and visualizations

## Requirements

Install the dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Quick start

### 1) Train the model

```bash
python automate_pipeline.py train \
  --train_degraded datasets/train/train/NoisyLR \
  --train_gt datasets/train/train/GT \
  --checkpoint_dir checkpoints \
  --epochs 300 \
  --batch_size 4
```

### 2) Evaluate on test data

```bash
python automate_pipeline.py evaluate \
  --input_dir datasets/Test_NoisyLR/NoisyLR \
  --output_dir outputs/restored_eval \
  --model_path checkpoints/best_model.pth
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
