"""
WEX 428 - Computer Vision Project
Module 5: Deblurring Module (Anti-Checkerboard & Smooth Natural Edges)

Optimizes a Deep Image Prior network to restore a blurred image using forward blur operator H:
min || H * f_theta(z) - y_blurred ||^2 + tv_weight * TV(f_theta(z))

Eliminates high-frequency null-space checkerboard artifacts via balanced Total Variation
and controlled gradient updates.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dip_models import build_dip_model, get_compute_device
from evaluation_module import evaluate_all_metrics


def _make_blur_conv_operator(kernel_2d, device):
    """Builds a fixed depthwise 2D convolution operator in PyTorch for an RGB image."""
    k = torch.from_numpy(kernel_2d).float().to(device)
    kh, kw = k.shape
    weight = k.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # shape (3, 1, kh, kw)
    pad_h, pad_w = kh // 2, kw // 2

    def apply_blur_op(x_tensor):
        x_pad = F.pad(x_tensor, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        return F.conv2d(x_pad, weight, groups=3)

    return apply_blur_op


def _total_variation_loss(x):
    """Isotropic Total Variation penalty to suppress high-frequency checkerboard ringing."""
    dh = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    dw = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return dh + dw


def run_dip_deblurring(
    blurred_rgb,
    blur_kernel,
    original_rgb=None,
    arch="unet",
    num_iterations=400,
    lr=0.01,
    tv_weight=3e-5,         # Balanced TV weight suppresses checkerboard grid without blurring edges
    in_channels=16,
    base_channels=32,
    ema_decay=0.90,         # Gentle EMA eliminates frame-to-frame pixel flicker
    log_every=25,
    snapshot_iters=(50, 150, 250, 400),
    progress_callback=None,
):
    device, device_name = get_compute_device()
    h, w, _ = blurred_rgb.shape

    # Moderate learning rate prevents high-frequency oscillation in deconvolution
    effective_lr = min(float(lr), 0.003)

    torch.manual_seed(42)
    model = build_dip_model(arch_name=arch, in_channels=in_channels, base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    criterion = nn.MSELoss()

    target_t = torch.from_numpy(np.transpose(blurred_rgb, (2, 0, 1))).unsqueeze(0).float().to(device)
    blur_op = _make_blur_conv_operator(blur_kernel, device)
    z_fixed = (torch.rand((1, in_channels, h, w), device=device) * 0.1).detach()

    out_ema = None
    best_psnr = -1.0
    best_img = None
    best_iter = 0
    history = []
    snapshots = {}

    start_time = time.time()

    for it in range(1, num_iterations + 1):
        optimizer.zero_grad()

        sharp_pred = model(z_fixed)
        reblurred_pred = blur_op(sharp_pred)

        recon_loss = criterion(reblurred_pred, target_t)
        tv_penalty = _total_variation_loss(sharp_pred) if tv_weight > 0 else 0.0
        loss = recon_loss + (tv_weight * tv_penalty)

        loss.backward()
        optimizer.step()

        curr_np = sharp_pred.detach().cpu().squeeze(0).numpy()
        curr_np = np.clip(np.transpose(curr_np, (1, 2, 0)), 0.0, 1.0).astype(np.float32)

        # Apply gentle EMA from iteration 50 onwards to smooth out any residual pixel noise
        if out_ema is None or it < 50:
            out_ema = curr_np.copy()
        else:
            out_ema = ema_decay * out_ema + (1.0 - ema_decay) * curr_np

        if it in snapshot_iters or it == num_iterations:
            snapshots[it] = out_ema.copy()

        if it == 1 or it % log_every == 0 or it == num_iterations:
            ref = original_rgb if original_rgb is not None else blurred_rgb
            metrics = evaluate_all_metrics(ref, out_ema)
            metrics["iteration"] = it
            metrics["loss"] = round(float(loss.item()), 6)
            history.append(metrics)

            if metrics["PSNR"] > best_psnr:
                best_psnr = metrics["PSNR"]
                best_img = out_ema.copy()
                best_iter = it

        if progress_callback is not None and (it % 10 == 0 or it == num_iterations):
            progress_callback(it, num_iterations, time.time() - start_time, out_ema)

    elapsed = round(time.time() - start_time, 2)
    final_img = out_ema if out_ema is not None else blurred_rgb
    if best_img is None:
        best_img = final_img

    ref = original_rgb if original_rgb is not None else blurred_rgb
    final_metrics = evaluate_all_metrics(ref, best_img)

    return {
        "restored_image": best_img,
        "final_iter_image": final_img,
        "best_iteration": best_iter,
        "num_iterations": num_iterations,
        "elapsed_time": elapsed,
        "device_used": device_name,
        "architecture": arch,
        "metrics": final_metrics,
        "degraded_metrics": evaluate_all_metrics(ref, blurred_rgb),
        "history": history,
        "snapshots": snapshots,
    }