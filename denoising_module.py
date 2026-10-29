"""
WEX 428 - Computer Vision Project
Module 4: Denoising Module (Optimized for Crisp Edge Sharpness)

Optimizes a randomly initialized Deep Image Prior network on a single noisy image
without any prior training data.
Tuned for high visual sharpness and fast convergence:
- Zero excessive input jitter (reg_noise_std=0) to preserve fine edge details
- Fast adaptive EMA that only smooths late iterations without blurring sharp structures
"""

import time
import numpy as np
import torch
import torch.nn as nn

from dip_models import build_dip_model, get_compute_device
from evaluation_module import evaluate_all_metrics


def _np_to_tensor(img_rgb, device):
    t = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).unsqueeze(0).float()
    return t.to(device)


def _tensor_to_np(tensor_4d):
    arr = tensor_4d.detach().cpu().squeeze(0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def run_dip_denoising(
    noisy_rgb,
    original_rgb=None,
    arch="unet",
    num_iterations=400,
    lr=0.015,
    reg_noise_std=0.0,      # Set to 0.0 to prevent artificial blurring of fine edges
    in_channels=16,
    base_channels=32,
    ema_decay=0.85,         # Light EMA (0.85) to suppress noise without softening sharp structures
    loss_type="mse",
    log_every=25,
    snapshot_iters=(50, 150, 250, 400),
    progress_callback=None,
):
    device, device_name = get_compute_device()
    h, w, _ = noisy_rgb.shape

    torch.manual_seed(42)
    model = build_dip_model(arch_name=arch, in_channels=in_channels, base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.L1Loss() if loss_type == "l1" else nn.MSELoss()

    target_t = _np_to_tensor(noisy_rgb, device)
    z_fixed = (torch.rand((1, in_channels, h, w), device=device) * 0.1).detach()

    out_smoothed = None
    best_psnr = -1.0
    best_img = None
    best_iter = 0
    history = []
    snapshots = {}

    start_time = time.time()

    for it in range(1, num_iterations + 1):
        optimizer.zero_grad()

        if reg_noise_std > 0:
            z_in = z_fixed + (torch.randn_like(z_fixed) * reg_noise_std)
        else:
            z_in = z_fixed

        out_t = model(z_in)
        loss = criterion(out_t, target_t)
        loss.backward()
        optimizer.step()

        curr_np = _tensor_to_np(out_t)

        # For the first 40% of iterations, track raw crisp network output directly.
        # Only apply light EMA smoothing in the later iterations to prevent noise fit.
        if it < int(num_iterations * 0.4):
            display_img = curr_np.copy()
            out_smoothed = curr_np.copy()
        else:
            out_smoothed = ema_decay * out_smoothed + (1.0 - ema_decay) * curr_np
            # Blend sharp instantaneous output with light smoothed output for crisp details
            display_img = 0.65 * curr_np + 0.35 * out_smoothed

        if it in snapshot_iters or it == num_iterations:
            snapshots[it] = display_img.copy()

        if it == 1 or it % log_every == 0 or it == num_iterations:
            ref = original_rgb if original_rgb is not None else noisy_rgb
            metrics = evaluate_all_metrics(ref, display_img)
            metrics["iteration"] = it
            metrics["loss"] = round(float(loss.item()), 6)
            history.append(metrics)

            if metrics["PSNR"] > best_psnr:
                best_psnr = metrics["PSNR"]
                best_img = display_img.copy()
                best_iter = it

        if progress_callback is not None and (it % 10 == 0 or it == num_iterations):
            progress_callback(it, num_iterations, time.time() - start_time, display_img)

    elapsed = round(time.time() - start_time, 2)
    final_img = display_img if display_img is not None else noisy_rgb
    if best_img is None:
        best_img = final_img

    ref = original_rgb if original_rgb is not None else noisy_rgb
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
        "degraded_metrics": evaluate_all_metrics(ref, noisy_rgb),
        "history": history,
        "snapshots": snapshots,
    }