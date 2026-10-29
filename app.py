"""
WEX 428 - Computer Vision Project
Live Testing Interface (Streamlit)

Interactive interface for the defense presentation live demo:
- Uses ONLY the reserved held-out test subset (500 images, never used during optimization)
- Automatically applies the selected image's assigned random degradation parameters
- Triggers live Deep Image Prior restoration with progress indicator and timer
- Displays Original, Degraded, and Restored images side by side with MSE, PSNR, SSIM
- Provides a download button for the restored image
"""

import io
import os
import csv
import numpy as np
from PIL import Image
import streamlit as st

from input_module import prepare_dataset, get_subset_records, load_image_rgb
from degradation_module import apply_assigned_degradation
from evaluation_module import evaluate_all_metrics
from denoising_module import run_dip_denoising
from deblurring_module import run_dip_deblurring
from visualization_module import plot_optimization_curves, plot_iteration_snapshots
from dip_models import get_compute_device

st.set_page_config(
    page_title="WEX 428 - Deep Image Prior Restoration",
    layout="wide",
)

st.title("WEX 428 - Deep Image Prior (DIP) Image Restoration System")
st.caption("Unsupervised Single-Image Denoising and Deblurring without Pre-trained Weights")

# Ensure dataset of 2,500+ images is initialized
manifest = prepare_dataset()
test_records = get_subset_records("test_live")
_, device_label = get_compute_device()

# Top Dataset Summary Banner
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("Total Dataset Size", f"{manifest['total_images']} images")
col_info2.metric("Train / Optimization Split", f"{manifest['train_opt_count']} images")
col_info3.metric("Reserved Live Test Split", f"{manifest['test_live_count']} images")
col_info4.metric("Compute Backend", device_label)

tab_live, tab_exp = st.tabs([
    "Live Testing Interface (Reserved Test Subset)",
    "Saved Experiment Comparisons",
])

with tab_live:
    st.subheader("Interactive Live Restoration on Held-Out Test Subset")

    ctrl_col, display_col = st.columns([1, 2.6])

    with ctrl_col:
        st.markdown("**1. Select Reserved Test Image**")
        options = [
            f"Test Image #{r['id']} ({r['filename']}) - {r['degradation']['default_task'].title()}"
            for r in test_records
        ]
        selected_idx = st.selectbox("Reserved Test Subset (500 held-out images):", range(len(options)), format_func=lambda i: options[i])
        selected_record = test_records[selected_idx]
        deg_params = selected_record["degradation"]

        st.markdown("**2. Assigned Random Degradation Parameters**")
        task_mode = st.radio(
            "Restoration Task Mode:",
            options=["Assigned Default", "Denoising", "Deblurring"],
            horizontal=True,
        )
        override = None
        if task_mode == "Denoising":
            override = "denoising"
        elif task_mode == "Deblurring":
            override = "deblurring"

        st.json(deg_params, expanded=False)

        st.markdown("**3. Deep Image Prior Hyperparameters**")
        arch_choice = st.selectbox(
            "Network Architecture:",
            options=[("unet", "U-Net (With Skip Connections)"), ("encoder_decoder", "Encoder-Decoder (No Skips)")],
            format_func=lambda x: x[1],
        )[0]

        res_size = st.select_slider("Working Image Resolution:", options=[64, 96, 128, 160], value=128)
        num_iters = st.slider("Optimization Iterations:", min_value=100, max_value=1000, value=350, step=50)
        learning_rate = st.select_slider("Learning Rate (Adam):", options=[0.005, 0.01, 0.015, 0.02], value=0.01)
        base_ch = st.select_slider("Base Filter Channels:", options=[16, 32, 48], value=32)

        run_btn = st.button("Start Deep Image Prior Restoration", type="primary", use_container_width=True)

    with display_col:
        orig_rgb = load_image_rgb(selected_record["filepath"], target_size=(res_size, res_size))
        degraded_rgb, blur_kernel, deg_desc = apply_assigned_degradation(orig_rgb, deg_params, task_override=override)
        deg_metrics = evaluate_all_metrics(orig_rgb, degraded_rgb)

        st.info(f"**Active Corruption on Selected Test Image:** {deg_desc}")

        img_col1, img_col2, img_col3 = st.columns(3)
        with img_col1:
            st.markdown("**Original Image**")
            st.image(orig_rgb, use_container_width=True)
            st.caption("Ground Truth Reference")

        with img_col2:
            st.markdown("**Degraded Image (Input)**")
            st.image(degraded_rgb, use_container_width=True)
            st.caption(
                f"MSE: `{deg_metrics['MSE']:.5f}` | "
                f"PSNR: `{deg_metrics['PSNR']:.2f} dB` | "
                f"SSIM: `{deg_metrics['SSIM']:.4f}`"
            )

        restored_placeholder = img_col3.empty()
        restored_caption = img_col3.empty()

        if not run_btn:
            with restored_placeholder.container():
                st.markdown("**Restored Image (DIP Output)**")
                st.image(degraded_rgb, use_container_width=True)
            restored_caption.caption("Click 'Start Deep Image Prior Restoration' to optimize network live.")
        else:
            progress_bar = st.progress(0, text="Initializing random DIP network parameters...")
            status_text = st.empty()

            def live_callback(curr_it, total_it, elapsed_s, current_img):
                frac = min(1.0, curr_it / float(total_it))
                progress_bar.progress(
                    frac,
                    text=f"Optimizing DIP Network: Iteration {curr_it}/{total_it} | Elapsed: {elapsed_s:.1f}s"
                )
                with restored_placeholder.container():
                    st.markdown("**Restored Image (Live Optimization)**")
                    st.image(current_img, use_container_width=True)

            active_task = override if override is not None else deg_params.get("default_task", "denoising")
            snap_list = sorted(list(set([
                max(25, num_iters // 4),
                max(50, num_iters // 2),
                max(75, (3 * num_iters) // 4),
                num_iters
            ])))

            if active_task == "denoising":
                loss_mode = "l1" if deg_params.get("noise_type") == "salt_and_pepper" else "mse"
                result = run_dip_denoising(
                    degraded_rgb,
                    original_rgb=orig_rgb,
                    arch=arch_choice,
                    num_iterations=num_iters,
                    lr=learning_rate,
                    base_channels=base_ch,
                    loss_type=loss_mode,
                    snapshot_iters=snap_list,
                    progress_callback=live_callback,
                )
            else:
                result = run_dip_deblurring(
                    degraded_rgb,
                    blur_kernel=blur_kernel,
                    original_rgb=orig_rgb,
                    arch=arch_choice,
                    num_iterations=num_iters,
                    lr=learning_rate,
                    base_channels=base_ch,
                    snapshot_iters=snap_list,
                    progress_callback=live_callback,
                )

            progress_bar.progress(1.0, text=f"Optimization Complete in {result['elapsed_time']}s (Early Stop Best Iteration: {result['best_iteration']})")
            res_m = result["metrics"]
            psnr_gain = round(res_m["PSNR"] - deg_metrics["PSNR"], 2)
            ssim_gain = round(res_m["SSIM"] - deg_metrics["SSIM"], 4)

            with restored_placeholder.container():
                st.markdown("**Restored Image (DIP Output)**")
                st.image(result["restored_image"], use_container_width=True)
            restored_caption.caption(
                f"MSE: `{res_m['MSE']:.5f}` | "
                f"PSNR: `{res_m['PSNR']:.2f} dB` ({psnr_gain:+.2f} dB) | "
                f"SSIM: `{res_m['SSIM']:.4f}` ({ssim_gain:+.4f})"
            )

            # Metrics Summary Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Final MSE", f"{res_m['MSE']:.5f}", f"{res_m['MSE'] - deg_metrics['MSE']:+.5f}", delta_color="inverse")
            m2.metric("Final PSNR (dB)", f"{res_m['PSNR']:.2f} dB", f"{psnr_gain:+.2f} dB")
            m3.metric("Final SSIM", f"{res_m['SSIM']:.4f}", f"{ssim_gain:+.4f}")
            m4.metric("Best Iteration", f"{result['best_iteration']} / {num_iters}")
            m5.metric("Processing Time", f"{result['elapsed_time']} s")

            # Download Restored Image Button
            pil_out = Image.fromarray(np.clip(result["restored_image"] * 255.0, 0, 255).astype(np.uint8))
            buf = io.BytesIO()
            pil_out.save(buf, format="PNG")
            st.download_button(
                label="Download Restored Image (.png)",
                data=buf.getvalue(),
                file_name=f"restored_img_{selected_record['id']}_{arch_choice}.png",
                mime="image/png",
            )

            st.markdown("#### Intermediate Optimization Snapshots")
            fig_snaps = plot_iteration_snapshots(result["snapshots"])
            if fig_snaps is not None:
                st.pyplot(fig_snaps)

            st.markdown("#### Iteration Convergence & Early Stopping Curve")
            fig_curve = plot_optimization_curves(result["history"], best_iteration=result["best_iteration"])
            if fig_curve is not None:
                st.pyplot(fig_curve)

with tab_exp:
    st.subheader("Experimental Analysis & Architecture Comparisons")
    csv_file = os.path.join("results", "experiment_summary_metrics.csv")
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        st.dataframe(reader, use_container_width=True)

        png_files = sorted([f for f in os.listdir("results") if f.endswith(".png")])
        for pf in png_files:
            st.markdown(f"**{pf}**")
            st.image(os.path.join("results", pf))
    else:
        st.info("Run `python experiment_module.py` (or click below) to generate the full suite of comparison tables and plots.")
        if st.button("Run Batch Experiments Now"):
            with st.spinner("Running experiments across noise levels, blur kernels, and architectures..."):
                from experiment_module import run_all_experiments
                run_all_experiments(num_iterations=250)
            st.rerun()
