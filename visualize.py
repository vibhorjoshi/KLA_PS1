import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from model import RestorationNet
from dataset import SemiconductorDataset


def visualize_results(model_path, degraded_path, gt_path, num_samples=3, out_path='outputs/visualization.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = RestorationNet().to(device)
    if not os.path.exists(model_path):
        print(f"Checkpoint not found at {model_path}, trying checkpoints/best_model.pth")
        alt = 'checkpoints/best_model.pth'
        if os.path.exists(alt):
            model_path = alt
        else:
            print('No checkpoint found. Aborting.')
            return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded: {model_path}")

    # Load dataset
    if not os.path.exists(degraded_path) or not os.path.exists(gt_path):
        print('Provided dataset paths not found, attempting to use local dataset paths under datasets/train/train')
        degraded_path = 'datasets/train/train/NoisyLR'
        gt_path = 'datasets/train/train/GT'
        if not os.path.exists(degraded_path) or not os.path.exists(gt_path):
            print('Fallback dataset paths not found. Aborting.')
            return

    dataset = SemiconductorDataset(degraded_path, gt_path, augment=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            x, y = dataset[i]
            x_in = x.unsqueeze(0).to(device)
            pred = model(x_in).squeeze(0).cpu().numpy()[0]

            img_noisy = x.numpy()[0]
            img_gt = y.numpy()[0]

            # Plotting
            axes[i, 0].imshow(img_noisy, cmap='gray')
            axes[i, 0].set_title(f"Sample {i+1}: Noisy Input")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(pred, cmap='gray')
            axes[i, 1].set_title(f"Sample {i+1}: Restored Output")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(img_gt, cmap='gray')
            axes[i, 2].set_title(f"Sample {i+1}: Ground Truth")
            axes[i, 2].axis('off')

    plt.savefig(out_path, bbox_inches='tight')
    print(f'Visualization saved to {out_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/model_epoch_50.pth')
    parser.add_argument('--degraded_path', type=str, default='/content/extracted_data/datasets/train/train/NoisyLR')
    parser.add_argument('--gt_path', type=str, default='/content/extracted_data/datasets/train/train/GT')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--out', type=str, default='outputs/visualization.png')
    args = parser.parse_args()
    visualize_results(args.model_path, args.degraded_path, args.gt_path, args.num_samples, args.out)
