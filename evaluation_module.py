"""
WEX 428 - Computer Vision Project
Module 6: Evaluation Module

Computes quantitative image restoration metrics:
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR in dB)
- Structural Similarity Index (SSIM)
"""

import numpy as np
from scipy.ndimage import uniform_filter


def compute_mse(original_rgb, test_rgb):
    """Calculates Mean Squared Error (MSE) between two [0, 1] RGB images."""
    orig = np.asarray(original_rgb, dtype=np.float64)
    test = np.asarray(test_rgb, dtype=np.float64)
    return float(np.mean((orig - test) ** 2))


def compute_psnr(original_rgb, test_rgb, data_range=1.0):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB)."""
    mse = compute_mse(original_rgb, test_rgb)
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_ssim(original_rgb, test_rgb, data_range=1.0):
    """
    Calculates Structural Similarity Index (SSIM) across RGB channels.
    Uses scikit-image if installed, otherwise uses standard windowed SSIM via scipy.
    """
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(
            original_rgb, test_rgb, channel_axis=-1, data_range=data_range
        ))
    except ImportError:
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        ssim_channels = []
        for ch in range(original_rgb.shape[-1]):
            x = original_rgb[..., ch].astype(np.float64)
            y = test_rgb[..., ch].astype(np.float64)
            mu_x = uniform_filter(x, size=7)
            mu_y = uniform_filter(y, size=7)
            mu_x_sq = mu_x ** 2
            mu_y_sq = mu_y ** 2
            mu_xy = mu_x * mu_y
            sigma_x_sq = uniform_filter(x * x, size=7) - mu_x_sq
            sigma_y_sq = uniform_filter(y * y, size=7) - mu_y_sq
            sigma_xy = uniform_filter(x * y, size=7) - mu_xy

            num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
            den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
            ssim_map = num / (den + 1e-12)
            ssim_channels.append(np.mean(ssim_map))
        return float(np.mean(ssim_channels))


def evaluate_all_metrics(original_rgb, test_rgb):
    """Returns a dictionary with MSE, PSNR, and SSIM rounded for display."""
    mse_val = compute_mse(original_rgb, test_rgb)
    psnr_val = compute_psnr(original_rgb, test_rgb)
    ssim_val = compute_ssim(original_rgb, test_rgb)
    return {
        "MSE": round(mse_val, 6),
        "PSNR": round(psnr_val, 2),
        "SSIM": round(ssim_val, 4),
    }
