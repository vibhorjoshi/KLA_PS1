# KLA SemiCon AI Hackathon 2026 — Image Restoration

**Team:** godlike  
**Institution:** Indian Institute of Information Technology Guwahati  
**Problem Statement:** PS01 — AI-Based Restoration of Degraded Images

---

## 1. Project Overview

This repository contains a deep learning-based image restoration pipeline for degraded semiconductor inspection images. The system is built around an RRDB-style residual dense architecture and is designed to recover high-quality grayscale images from low-resolution, noisy inputs stored as `.npy` arrays.

The project targets the following degradations:
- **Speckle noise / out-of-range pixel intensity**
- **Gaussian blur and additive noise**
- **2× spatial resolution degradation**

The model takes a noisy low-resolution grayscale array of shape `(1, H, W)` and reconstructs a clean, upscaled image of shape `(1, 2H, 2W)`.

**Key specifications:**
- Architecture: RRDB-Net with residual dense blocks
- Parameters: ~2.8M
- Typical inference speed: ~2.07 ms/image on NVIDIA H100
- Input type: grayscale `.npy` with shape `(1, H, W)`
- Output type: restored `.npy` with shape `(1, 2H, 2W)`
- Output range: clamped to `[0, 1]`

---

## 2. Repository Structure

```text
KLA PROJECT/
├── README.md                     # Project documentation
├── requirements.txt              # Frozen dependencies for reproducibility
├── automate_pipeline.py          # Unified automation entry point
├── train.py                      # Training script
├── evaluate.py                   # Standalone inference / benchmark script
├── test_metrics.py               # PSNR / SSIM metric evaluation
├── compare_and_evaluate.py       # Side-by-side comparison PNG generation
├── visualize.py                  # Visualization helper
├── dataset.py                    # Data loading and preprocessing
├── model.py                      # RRDB-based restoration network
├── losses.py                     # Loss definitions
├── utils.py                      # Utility functions and metrics helpers
├── compute_metrics.py            # Metric calculations
├── checkpoints/
│   ├── best_model.pth            # Best trained checkpoint
│   ├── checkpoint_epoch_1.pth    # Intermediate checkpoint
│   ├── checkpoint_epoch_2.pth    # Intermediate checkpoint
│   └── final_model.pth           # Final training checkpoint
├── datasets/
│   ├── train/
│   │   └── train/
│   │       ├── NoisyLR/
│   │       └── GT/
│   └── Test_NoisyLR/
│       └── NoisyLR/
├── outputs/
│   └── restored_eval/            # Restored evaluation outputs
├── .gitignore
├── LICENSE
└── __pycache__/                 # Local Python cache (not committed)
```

---

## 3. Setup Instructions

### 3.1 Clone the repository

```bash
git clone https://github.com/vibhorjoshi/KLA_PS1.git
cd KLA_PS1
```

### 3.2 Create a virtual environment

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3.3 Install dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `torch==2.13.0`
- `torchvision==0.28.0`
- `numpy==2.5.1`
- `Pillow==12.3.0`
- `scikit-image==0.26.0`
- `matplotlib==3.11.1`
- `scipy==1.18.0`
- `tqdm==4.70.0`

---

## 4. Trained Model Weights

The project already includes trained weights under:

```text
checkpoints/best_model.pth
```

This checkpoint is the default model used for benchmark and evaluation runs.

---

## 5. Evaluation / Inference

The primary benchmark script is `evaluate.py`. It scans a directory of degraded `.npy` files, runs the trained model, and saves corresponding restored outputs.

### 5.1 Usage

```bash
python evaluate.py \
  --input_dir ./datasets/Test_NoisyLR/NoisyLR \
  --output_dir ./outputs/restored_eval \
  --model_path ./checkpoints/best_model.pth
```

### 5.2 Windows PowerShell example

```powershell
python evaluate.py `
  --input_dir ".\datasets\Test_NoisyLR\NoisyLR" `
  --output_dir ".\outputs\restored_eval" `
  --model_path ".\checkpoints\best_model.pth"
```

### 5.3 What the script does

1. Loads all `.npy` inputs from the input folder
2. Runs model inference on GPU if available, otherwise CPU
3. Clamps output intensities to `[0, 1]`
4. Saves restored arrays to the output directory
5. Prints per-image processing time and average throughput


---

## 6. Training from Scratch

To reproduce training, use the paired dataset organized as follows:

```text
datasets/
└── train/
    └── train/
        ├── NoisyLR/
        │   ├── 000000.npy
        │   └── ...
        └── GT/
            ├── 000000.npy
            └── ...
```

### 6.1 Run training

```bash
python train.py \
  --train_degraded ./datasets/train/train/NoisyLR \
  --train_gt ./datasets/train/train/GT \
  --checkpoint_dir ./checkpoints \
  --epochs 300 \
  --batch_size 4 \
  --lr 2e-4
```

**Notes:**
- Windows users can keep the default `num_workers=0`
- Mixed precision is enabled automatically when CUDA is available
- Best checkpoint is selected based on validation quality

### 6.2 Resume training

Use the same command with the same dataset and output directory, or load the latest checkpoint in the training script if resuming an interrupted run.

---

## 7. Automated Pipeline

The repository also includes a unified automation wrapper that supports the major stages of the workflow.

```bash
python automate_pipeline.py --help
```

Available actions include:
- `train`
- `evaluate`
- `visualize`
- `metrics`
- `full`

Example:

```bash
python automate_pipeline.py full \
  --epochs 300 \
  --batch_size 4 \
  --lr 2e-4
```

---

## 8. Computing Test Metrics

If ground-truth images exist for the evaluation set, run:

```bash
python test_metrics.py \
  --restored_dir ./outputs/restored_eval \
  --gt_dir ./path/to/test/ground_truth \
  --output_csv ./metrics.csv
```

This writes a CSV report containing per-image and aggregate PSNR/SSIM values.

---

## 9. Generating Visual Comparisons

To create side-by-side degraded/restored/ground-truth comparisons for reporting or presentation, use:

```bash
python compare_and_evaluate.py \
  --degraded_dir ./datasets/Test_NoisyLR/NoisyLR \
  --restored_dir ./outputs/restored_eval \
  --gt_dir ./path/to/test/ground_truth \
  --output_dir ./evaluation_results
```

This generates:
- `evaluation_results/comparisons/*.png` — side-by-side image panels
- `evaluation_results/results.csv` — PSNR/SSIM summary table

---

## 10. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 / Linux | Ubuntu 22.04 or newer |
| Python | 3.9+ | 3.10+ |
| RAM | 8 GB | 16 GB |
| GPU | CPU acceptable | NVIDIA RTX / A100 / H100 |
| CUDA | Not required | 12.1+ |
| Storage | 2 GB | 5 GB |

---

## 11. Results Summary

| Metric | Value |
|--------|-------|
| Mean PSNR | 22.62 dB |
| Mean SSIM | 0.6275 |
| Inference Time | ~2.07 ms/image on H100 |
| Parameters | ~2.8M |
| Checkpoint Size | ~17.4 MB |

---

## 12. Submission Checklist

This repository satisfies the core hackathon submission requirements:
- [x] Training pipeline available
- [x] Inference / evaluation script available
- [x] Pretrained model weights included
- [x] Dependencies pinned in `requirements.txt`
- [x] Output restoration examples generated
- [x] Documentation available in `README.md`

---

## 13. Contact

For questions related to the project, dataset preparation, or reproduction steps, please open an issue in the repository or contact the team through the project repository.

**Team:** godlike  
**Institution:** Indian Institute of Information Technology Guwahati

---

## 14. References

1. T. Kumar et al., "Image Data Augmentation Approaches: A Comprehensive Survey," *IEEE Access*, 2024.
2. L. Zhai et al., "A comprehensive review of deep learning-based real-world image restoration," *IEEE Access*, 2023.
3. J. Terven et al., "A comprehensive survey of loss functions and metrics in deep learning," *Artificial Intelligence Review*, 2025.
4. V. Monga et al., "Algorithm Unrolling," *IEEE Signal Processing Magazine*, 2021.
6. KLA Corporation, "SemiCon AI Hackathon 2026 — PS01," 2026.

