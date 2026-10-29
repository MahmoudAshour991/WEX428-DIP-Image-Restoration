"""
WEX 428 - Computer Vision Project
Module 7: Visualization Module

Generates visual comparison figures for the report and presentation:
- Side-by-side comparison: Original Image, Degraded Image, Restored Image
- Optimization iteration snapshots showing progression and early stopping
- PSNR and SSIM curves across optimization iterations
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_restoration_triplet(original_rgb, degraded_rgb, restored_rgb, result_info, save_path=None):
    """
    Plots Original, Degraded, and Restored images side by side with MSE, PSNR, and SSIM values.
    """
    deg_m = result_info.get("degraded_metrics", {})
    res_m = result_info.get("metrics", {})
    arch = result_info.get("architecture", "unet").upper()
    best_it = result_info.get("best_iteration", result_info.get("num_iterations", 0))
    elapsed = result_info.get("elapsed_time", 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=140)

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Ground Truth", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(degraded_rgb)
    axes[1].set_title(
        f"Degraded Input\nMSE: {deg_m.get('MSE', 0):.4f} | PSNR: {deg_m.get('PSNR', 0):.2f} dB | SSIM: {deg_m.get('SSIM', 0):.3f}",
        fontsize=9.5,
    )
    axes[1].axis("off")

    axes[2].imshow(restored_rgb)
    axes[2].set_title(
        f"DIP Restored ({arch}, Iter {best_it}, {elapsed}s)\n"
        f"MSE: {res_m.get('MSE', 0):.4f} | PSNR: {res_m.get('PSNR', 0):.2f} dB | SSIM: {res_m.get('SSIM', 0):.3f}",
        fontsize=9.5,
        color="#1b5e20",
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_iteration_snapshots(snapshots_dict, save_path=None):
    """Plots intermediate restored images across optimization iterations."""
    iters = sorted(snapshots_dict.keys())
    n = len(iters)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), dpi=130)
    if n == 1:
        axes = [axes]

    for ax, it in zip(axes, iters):
        ax.imshow(snapshots_dict[it])
        ax.set_title(f"Iteration {it}", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_optimization_curves(history, best_iteration=None, title="DIP Optimization Progress", save_path=None):
    """Plots PSNR (dB) and SSIM across iterations and marks the Early Stopping point."""
    if not history:
        return None

    iters = [h["iteration"] for h in history]
    psnrs = [h["PSNR"] for h in history]
    ssims = [h["SSIM"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=130)

    ax1.plot(iters, psnrs, marker="o", markersize=3.5, linewidth=1.8, color="#1565c0", label="PSNR (dB)")
    if best_iteration is not None:
        ax1.axvline(best_iteration, color="#d32f2f", linestyle="--", label=f"Early Stop (Iter {best_iteration})")
    ax1.set_title(f"{title} - PSNR", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("PSNR (dB)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8.5)

    ax2.plot(iters, ssims, marker="s", markersize=3.5, linewidth=1.8, color="#2e7d32", label="SSIM")
    if best_iteration is not None:
        ax2.axvline(best_iteration, color="#d32f2f", linestyle="--", label=f"Early Stop (Iter {best_iteration})")
    ax2.set_title(f"{title} - SSIM", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("SSIM")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
