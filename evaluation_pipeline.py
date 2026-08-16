#!/usr/bin/env python3
"""Automation entry point for the KLA semiconductor restoration pipeline.

Examples:
    python automate_pipeline.py train --train_degraded datasets/train/train/NoisyLR --train_gt datasets/train/train/GT
    python automate_pipeline.py evaluate --input_dir datasets/Test_NoisyLR/NoisyLR --output_dir outputs/restored_eval --model_path checkpoints/best_model.pth
    python automate_pipeline.py full --train_degraded datasets/train/train/NoisyLR --train_gt datasets/train/train/GT --input_dir datasets/Test_NoisyLR/NoisyLR --output_dir outputs/restored_eval
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_script(script_name: str, args: list[str]) -> None:
    command = [PYTHON, str(ROOT / script_name), *args]
    print(f"\n>>> {' '.join(command)}")
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def cmd_train(args: argparse.Namespace) -> None:
    run_script(
        "train.py",
        [
            "--train_degraded", args.train_degraded,
            "--train_gt", args.train_gt,
            "--checkpoint_dir", args.checkpoint_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--log_interval", str(args.log_interval),
            "--save_interval", str(args.save_interval),
        ],
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    run_script(
        "evaluate.py",
        [
            "--input_dir", args.input_dir,
            "--output_dir", args.output_dir,
            "--model_path", args.model_path,
        ],
    )


def cmd_visualize(args: argparse.Namespace) -> None:
    run_script(
        "visualize.py",
        [
            "--model_path", args.model_path,
            "--degraded_path", args.degraded_path,
            "--gt_path", args.gt_path,
            "--num_samples", str(args.num_samples),
            "--out", args.out,
        ],
    )


def cmd_metrics(args: argparse.Namespace) -> None:
    run_script(
        "compute_metrics.py",
        [],
    )


def cmd_full(args: argparse.Namespace) -> None:
    cmd_train(args)
    cmd_evaluate(args)
    cmd_visualize(args)
    cmd_metrics(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automate training, evaluation, visualization, and metric computation for the restoration project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train the restoration model")
    train_p.add_argument("--train_degraded", type=str, default="datasets/train/train/NoisyLR", help="Folder containing degraded input .npy files")
    train_p.add_argument("--train_gt", type=str, default="datasets/train/train/GT", help="Folder containing matching GT .npy files")
    train_p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    train_p.add_argument("--epochs", type=int, default=300)
    train_p.add_argument("--batch_size", type=int, default=4)
    train_p.add_argument("--lr", type=float, default=2e-4)
    train_p.add_argument("--log_interval", type=int, default=10)
    train_p.add_argument("--save_interval", type=int, default=50)
    train_p.set_defaults(func=cmd_train)

    eval_p = subparsers.add_parser("evaluate", help="Run model inference on a folder of degraded images")
    eval_p.add_argument("--input_dir", type=str, default="datasets/Test_NoisyLR/NoisyLR", help="Folder with degraded .npy files")
    eval_p.add_argument("--output_dir", type=str, default="outputs/restored_eval", help="Folder to save reconstructed .npy files")
    eval_p.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Checkpoint path for the trained model")
    eval_p.set_defaults(func=cmd_evaluate)

    vis_p = subparsers.add_parser("visualize", help="Create a comparison grid of noisy, restored, and GT images")
    vis_p.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    vis_p.add_argument("--degraded_path", type=str, default="datasets/train/train/NoisyLR")
    vis_p.add_argument("--gt_path", type=str, default="datasets/train/train/GT")
    vis_p.add_argument("--num_samples", type=int, default=3)
    vis_p.add_argument("--out", type=str, default="outputs/visualization.png")
    vis_p.set_defaults(func=cmd_visualize)

    metrics_p = subparsers.add_parser("metrics", help="Compute PSNR and SSIM for saved restoration outputs")
    metrics_p.set_defaults(func=cmd_metrics)

    full_p = subparsers.add_parser("full", help="Run the full training + evaluation + visualization + metric flow")
    full_p.add_argument("--train_degraded", type=str, default="datasets/train/train/NoisyLR")
    full_p.add_argument("--train_gt", type=str, default="datasets/train/train/GT")
    full_p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    full_p.add_argument("--epochs", type=int, default=300)
    full_p.add_argument("--batch_size", type=int, default=4)
    full_p.add_argument("--lr", type=float, default=2e-4)
    full_p.add_argument("--log_interval", type=int, default=10)
    full_p.add_argument("--save_interval", type=int, default=50)
    full_p.add_argument("--input_dir", type=str, default="datasets/Test_NoisyLR/NoisyLR")
    full_p.add_argument("--output_dir", type=str, default="outputs/restored_eval")
    full_p.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    full_p.add_argument("--degraded_path", type=str, default="datasets/train/train/NoisyLR")
    full_p.add_argument("--gt_path", type=str, default="datasets/train/train/GT")
    full_p.add_argument("--num_samples", type=int, default=3)
    full_p.add_argument("--out", type=str, default="outputs/visualization.png")
    full_p.set_defaults(func=cmd_full)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
