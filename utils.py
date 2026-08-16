import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR in dB."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))

def calculate_ssim(img1, img2, data_range=1.0):
    """
    img1, img2: torch tensors (B, C, H, W) or (C, H, W)
    Returns average SSIM across batch
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
        
    # Handle batch dimension
    if img1.ndim == 4:
        ssims = []
        for i in range(img1.shape[0]):
            ssims.append(
                ssim_func(
                    img1[i, 0], img2[i, 0],
                    data_range=data_range
                )
            )
        return np.mean(ssims)
    else:
        return ssim_func(img1[0], img2[0], data_range=data_range)

def normalize_input(img):
    mean = img.mean()
    std = img.std() + 1e-8
    return (img - mean) / std, mean, std

def denormalize_output(img, mean, std):
    return img * std + mean
