"""
WEX 428 - Computer Vision Project
Module 8: Experiment and Comparison Module

Runs systematic experiments on the training/optimization subset (never touching
the reserved test_live subset) to compare:
1. Different noise levels (Gaussian sigma = 15, 25, 50 & Salt-and-Pepper = 5%, 10%)
2. Different blur levels (Gaussian kernel = 5x5, 9x9 & Motion blur = 7x7, 11x11)
3. Different neural network architectures (UNetDIP vs. EncoderDecoderDIP)
4. Different optimization iterations & Early Stopping behavior

Saves summary CSV tables and comparison figures to the 'results/' directory.
"""

import os
import csv
import matplotlib.pyplot as plt

from input_module import get_subset_records, load_image_rgb
from degradation_module import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    apply_gaussian_blur,
    apply_motion_blur,
)
from denoising_module import run_dip_denoising
from deblurring_module import run_dip_deblurring
from visualization_module import (
    plot_restoration_triplet,
    plot_iteration_snapshots,
    plot_optimization_curves,
)

RESULTS_DIR = "results"


def _save_csv(filepath, rows, fieldnames):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_all_experiments(num_iterations=350, img_size=(128, 128)):
    """
    Runs the 4 core experimental suites on sample images from 'train_opt'
    and exports CSV tables + figures into 'results/'.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_records = get_subset_records("train_opt")
    sample_rec = train_records[0]
    orig_img = load_image_rgb(sample_rec["filepath"], target_size=img_size)

    summary_rows = []

    # Experiment 1: Different Noise Levels & Types
    print("=== Experiment 1: Comparing Different Noise Levels ===")
    noise_configs = [
        ("Gaussian sigma=15", lambda x: add_gaussian_noise(x, sigma=15.0, seed=101), "mse"),
        ("Gaussian sigma=25", lambda x: add_gaussian_noise(x, sigma=25.0, seed=102), "mse"),
        ("Gaussian sigma=50", lambda x: add_gaussian_noise(x, sigma=50.0, seed=103), "mse"),
        ("Salt & Pepper p=0.05", lambda x: add_salt_and_pepper_noise(x, amount=0.05, seed=104), "l1"),
        ("Salt & Pepper p=0.10", lambda x: add_salt_and_pepper_noise(x, amount=0.10, seed=105), "l1"),
    ]

    for label, deg_fn, loss_mode in noise_configs:
        noisy = deg_fn(orig_img)
        res = run_dip_denoising(
            noisy, original_rgb=orig_img, arch="unet",
            num_iterations=num_iterations, loss_type=loss_mode
        )
        safe_name = label.lower().replace(" ", "_").replace("=", "").replace("&", "and").replace(".", "")
        plot_restoration_triplet(
            orig_img, noisy, res["restored_image"], res,
            save_path=os.path.join(RESULTS_DIR, f"exp1_noise_{safe_name}.png")
        )
        plt.close("all")
        summary_rows.append({
            "Experiment": "Noise Levels",
            "Setting": label,
            "Architecture": "UNetDIP",
            "Degraded_MSE": res["degraded_metrics"]["MSE"],
            "Degraded_PSNR": res["degraded_metrics"]["PSNR"],
            "Degraded_SSIM": res["degraded_metrics"]["SSIM"],
            "Restored_MSE": res["metrics"]["MSE"],
            "Restored_PSNR": res["metrics"]["PSNR"],
            "Restored_SSIM": res["metrics"]["SSIM"],
            "Best_Iteration": res["best_iteration"],
            "Time_sec": res["elapsed_time"],
        })

    # Experiment 2: Different Blur Levels & Types
    print("=== Experiment 2: Comparing Different Blur Levels ===")
    blur_configs = [
        ("Gaussian Blur 5x5 (sigma=1.2)", lambda x: apply_gaussian_blur(x, kernel_size=5, sigma=1.2)),
        ("Gaussian Blur 9x9 (sigma=2.0)", lambda x: apply_gaussian_blur(x, kernel_size=9, sigma=2.0)),
        ("Motion Blur 7x7 (angle=45)", lambda x: apply_motion_blur(x, kernel_size=7, angle=45.0)),
        ("Motion Blur 11x11 (angle=30)", lambda x: apply_motion_blur(x, kernel_size=11, angle=30.0)),
    ]

    for label, blur_fn in blur_configs:
        blurred, kernel = blur_fn(orig_img)
        res = run_dip_deblurring(
            blurred, kernel, original_rgb=orig_img, arch="unet",
            num_iterations=num_iterations
        )
        safe_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        plot_restoration_triplet(
            orig_img, blurred, res["restored_image"], res,
            save_path=os.path.join(RESULTS_DIR, f"exp2_blur_{safe_name}.png")
        )
        plt.close("all")
        summary_rows.append({
            "Experiment": "Blur Levels",
            "Setting": label,
            "Architecture": "UNetDIP",
            "Degraded_MSE": res["degraded_metrics"]["MSE"],
            "Degraded_PSNR": res["degraded_metrics"]["PSNR"],
            "Degraded_SSIM": res["degraded_metrics"]["SSIM"],
            "Restored_MSE": res["metrics"]["MSE"],
            "Restored_PSNR": res["metrics"]["PSNR"],
            "Restored_SSIM": res["metrics"]["SSIM"],
            "Best_Iteration": res["best_iteration"],
            "Time_sec": res["elapsed_time"],
        })

    # Experiment 3 & 4: Architecture Comparison (UNetDIP vs EncoderDecoderDIP) & Early Stopping
    print("=== Experiment 3 & 4: Comparing Neural Network Architectures & Iterations ===")
    noisy_test = add_gaussian_noise(orig_img, sigma=25.0, seed=202)
    for arch_code, arch_label in [("unet", "UNetDIP (With Skip)"), ("encoder_decoder", "EncoderDecoderDIP (No Skip)")]:
        res = run_dip_denoising(
            noisy_test, original_rgb=orig_img, arch=arch_code,
            num_iterations=max(num_iterations, 500),
            snapshot_iters=(50, 150, 300, 500),
        )
        plot_restoration_triplet(
            orig_img, noisy_test, res["restored_image"], res,
            save_path=os.path.join(RESULTS_DIR, f"exp3_arch_{arch_code}_triplet.png")
        )
        plot_iteration_snapshots(
            res["snapshots"],
            save_path=os.path.join(RESULTS_DIR, f"exp4_iters_{arch_code}_snapshots.png")
        )
        plot_optimization_curves(
            res["history"], best_iteration=res["best_iteration"],
            title=f"{arch_label}",
            save_path=os.path.join(RESULTS_DIR, f"exp5_earlystop_{arch_code}_curves.png")
        )
        plt.close("all")
        summary_rows.append({
            "Experiment": "Architecture & Early Stopping",
            "Setting": "Gaussian sigma=25",
            "Architecture": arch_label,
            "Degraded_MSE": res["degraded_metrics"]["MSE"],
            "Degraded_PSNR": res["degraded_metrics"]["PSNR"],
            "Degraded_SSIM": res["degraded_metrics"]["SSIM"],
            "Restored_MSE": res["metrics"]["MSE"],
            "Restored_PSNR": res["metrics"]["PSNR"],
            "Restored_SSIM": res["metrics"]["SSIM"],
            "Best_Iteration": res["best_iteration"],
            "Time_sec": res["elapsed_time"],
        })

    csv_path = os.path.join(RESULTS_DIR, "experiment_summary_metrics.csv")
    _save_csv(csv_path, summary_rows, list(summary_rows[0].keys()))
    print(f"All experiments completed. Metrics saved to '{csv_path}'.")
    return summary_rows


if __name__ == "__main__":
    run_all_experiments(num_iterations=350)
