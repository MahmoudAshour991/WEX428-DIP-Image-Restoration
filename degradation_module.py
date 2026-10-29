"""
WEX 428 - Computer Vision Project
Module 2: Image Degradation Module

Implements artificial image degradations:
- Gaussian Noise
- Salt-and-Pepper Noise
- Gaussian Blur
- Motion Blur
Also applies the per-image randomly assigned degradation from the dataset manifest.
"""

import numpy as np
import cv2


def add_gaussian_noise(img_rgb, sigma=25.0, seed=None):
    """
    Adds zero-mean Gaussian noise with standard deviation sigma (on 0-255 scale)
    to a normalized [0, 1] RGB image.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    sigma_norm = float(sigma) / 255.0
    noise = rng.normal(loc=0.0, scale=sigma_norm, size=img_rgb.shape).astype(np.float32)
    noisy = np.clip(img_rgb + noise, 0.0, 1.0)
    return noisy


def add_salt_and_pepper_noise(img_rgb, amount=0.05, salt_vs_pepper=0.5, seed=None):
    """
    Adds Salt-and-Pepper impulse noise to a normalized [0, 1] RGB image.
    amount: fraction of pixels corrupted (e.g., 0.05 = 5% of pixels).
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    noisy = img_rgb.copy()
    h, w, _ = noisy.shape

    # Salt (white pixels = 1.0)
    num_salt = int(np.ceil(amount * h * w * salt_vs_pepper))
    ys = rng.randint(0, h, num_salt)
    xs = rng.randint(0, w, num_salt)
    noisy[ys, xs, :] = 1.0

    # Pepper (black pixels = 0.0)
    num_pepper = int(np.ceil(amount * h * w * (1.0 - salt_vs_pepper)))
    yp = rng.randint(0, h, num_pepper)
    xp = rng.randint(0, w, num_pepper)
    noisy[yp, xp, :] = 0.0

    return noisy


def make_gaussian_kernel(kernel_size=7, sigma=1.6):
    """Creates a normalized 2D Gaussian blur kernel of shape (kernel_size, kernel_size)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * (sigma ** 2)))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def make_motion_kernel(kernel_size=9, angle=45.0):
    """Creates a normalized 2D linear motion blur kernel at the specified angle in degrees."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)

    for i in range(-center, center + 1):
        x = int(round(center + i * dx))
        y = int(round(center + i * dy))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    if np.sum(kernel) == 0:
        kernel[center, :] = 1.0
    kernel /= np.sum(kernel)
    return kernel


def apply_blur_kernel(img_rgb, kernel):
    """Convolves an RGB [0, 1] image with a 2D blur kernel using reflective padding."""
    blurred = cv2.filter2D(img_rgb, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def apply_gaussian_blur(img_rgb, kernel_size=7, sigma=1.6):
    """Applies Gaussian blur and returns (blurred_img, kernel)."""
    kernel = make_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    blurred = apply_blur_kernel(img_rgb, kernel)
    return blurred, kernel


def apply_motion_blur(img_rgb, kernel_size=9, angle=45.0):
    """Applies linear motion blur and returns (blurred_img, kernel)."""
    kernel = make_motion_kernel(kernel_size=kernel_size, angle=angle)
    blurred = apply_blur_kernel(img_rgb, kernel)
    return blurred, kernel


def apply_assigned_degradation(img_rgb, deg_info, task_override=None):
    """
    Applies the per-image random degradation stored in the dataset manifest.
    Returns (degraded_img, kernel_or_none, summary_text).
    """
    task = task_override if task_override is not None else deg_info.get("default_task", "denoising")
    seed = deg_info.get("seed", 42)

    if task == "denoising":
        n_type = deg_info.get("noise_type", "gaussian")
        if n_type == "gaussian":
            sigma = deg_info.get("gaussian_sigma", 25.0)
            degraded = add_gaussian_noise(img_rgb, sigma=sigma, seed=seed)
            desc = f"Denoising | Gaussian Noise (sigma={sigma})"
        else:
            amt = deg_info.get("sp_amount", 0.05)
            degraded = add_salt_and_pepper_noise(img_rgb, amount=amt, seed=seed)
            desc = f"Denoising | Salt & Pepper Noise (amount={amt})"
        return degraded, None, desc
    else:
        b_type = deg_info.get("blur_type", "gaussian")
        k_size = deg_info.get("blur_kernel_size", 7)
        if b_type == "gaussian":
            sigma = deg_info.get("blur_sigma", 1.6)
            degraded, kernel = apply_gaussian_blur(img_rgb, kernel_size=k_size, sigma=sigma)
            desc = f"Deblurring | Gaussian Blur (kernel={k_size}x{k_size}, sigma={sigma})"
        else:
            angle = deg_info.get("motion_angle", 45)
            degraded, kernel = apply_motion_blur(img_rgb, kernel_size=k_size, angle=angle)
            desc = f"Deblurring | Motion Blur (kernel={k_size}x{k_size}, angle={angle} deg)"
        return degraded, kernel, desc
