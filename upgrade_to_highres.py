"""
Script to upgrade the dataset from low-res 32x32 CIFAR-10 to high-resolution,
crisp natural images (Flowers-102 / Caltech / custom folder) so images look
sharp and professional in the Streamlit Live Demo and presentation.
"""

import os
import shutil
import json
import random
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "images")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "dataset_manifest.json")
TOTAL_IMAGES = 2500
TRAIN_RATIO = 0.80


def generate_random_degradation_params(img_id, seed=42):
    rng = random.Random(seed + img_id * 17)
    task_type = rng.choice(["denoising", "deblurring"])
    noise_type = rng.choice(["gaussian", "salt_and_pepper"])
    gaussian_sigma = rng.choice([15.0, 25.0, 35.0, 50.0])
    sp_amount = rng.choice([0.03, 0.05, 0.08, 0.12])

    blur_type = rng.choice(["gaussian", "motion"])
    # 5x5 to 9x9 is ideal for 128x128/256x256 images without completely destroying them
    blur_kernel_size = rng.choice([5, 7, 9])
    blur_sigma = round(rng.uniform(1.2, 2.2), 2)
    motion_angle = rng.choice([0, 30, 45, 60, 90, 135])

    return {
        "image_id": img_id,
        "seed": seed + img_id * 17,
        "default_task": task_type,
        "noise_type": noise_type,
        "gaussian_sigma": gaussian_sigma,
        "sp_amount": sp_amount,
        "blur_type": blur_type,
        "blur_kernel_size": blur_kernel_size,
        "blur_sigma": blur_sigma,
        "motion_angle": motion_angle,
    }


def download_and_build_highres(target_size=256):
    print("=" * 60)
    print("UPGRADING DATASET TO HIGH-RESOLUTION CRISP NATURAL IMAGES")
    print("=" * 60)

    # 1. Clear old low-res CIFAR-10 images
    if os.path.exists(DATASET_DIR):
        print(f"Clearing old low-res images in '{DATASET_DIR}'...")
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR, exist_ok=True)
    if os.path.exists(MANIFEST_PATH):
        os.remove(MANIFEST_PATH)

    saved_count = 0

    # 2. Try downloading Oxford Flowers-102 (8,189 high-resolution images, ~330MB)
    try:
        import torchvision
        raw_cache = os.path.join(BASE_DIR, "data", "torchvision_raw")
        os.makedirs(raw_cache, exist_ok=True)
        print("Downloading Oxford Flowers-102 high-resolution dataset via torchvision...")
        print("(8,000+ crisp high-res natural images with rich colors and sharp edges)...")
        ds = torchvision.datasets.Flowers102(root=raw_cache, split="train", download=True)
        ds_test = torchvision.datasets.Flowers102(root=raw_cache, split="test", download=True)
        combined = [ds[i] for i in range(len(ds))] + [ds_test[i] for i in range(len(ds_test))]
        print(f"Loaded {len(combined)} raw high-res images. Extracting {TOTAL_IMAGES} images...")

        rng = random.Random(42)
        rng.shuffle(combined)

        for pil_img, label in combined:
            if saved_count >= TOTAL_IMAGES:
                break
            # Crop to square and resize with high-quality Lanczos resampling
            w, h = pil_img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            cropped = pil_img.crop((left, top, left + min_dim, top + min_dim))
            resized = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

            fname = f"highres_{saved_count:04d}_class{label}.png"
            resized.save(os.path.join(DATASET_DIR, fname), quality=95)
            saved_count += 1
            if saved_count % 500 == 0:
                print(f"  Processed {saved_count}/{TOTAL_IMAGES} high-res images...")
        print(f"Successfully saved {saved_count} crisp high-res images!")
    except Exception as e:
        print(f"Flowers-102 download notice: {e}")

    # Fallback to procedural high-frequency patterns if offline
    if saved_count < TOTAL_IMAGES:
        import numpy as np
        print(f"Generating remaining {TOTAL_IMAGES - saved_count} procedural images at {target_size}x{target_size}...")
        while saved_count < TOTAL_IMAGES:
            rng = np.random.RandomState(saved_count + 5000)
            x = np.linspace(0, 1, target_size)
            y = np.linspace(0, 1, target_size)
            xv, yv = np.meshgrid(x, y)
            img = np.zeros((target_size, target_size, 3), dtype=np.float32)

            for _ in range(8):
                cx, cy = rng.uniform(0.1, 0.9, 2)
                r = rng.uniform(0.08, 0.25)
                color = rng.uniform(0.1, 0.9, 3)
                mask = (xv - cx) ** 2 + (yv - cy) ** 2 <= r ** 2
                img[mask] = color

            fname = f"highres_{saved_count:04d}.png"
            pil_out = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
            pil_out.save(os.path.join(DATASET_DIR, fname))
            saved_count += 1

    # 3. Create new dataset manifest with 80/20 train/test split
    all_files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])[:TOTAL_IMAGES]
    rng = random.Random(42)
    shuffled_idx = list(range(len(all_files)))
    rng.shuffle(shuffled_idx)

    num_train = int(len(all_files) * TRAIN_RATIO)
    train_ids = set(shuffled_idx[:num_train])

    records = []
    for i, fname in enumerate(all_files):
        subset = "train_opt" if i in train_ids else "test_live"
        deg_params = generate_random_degradation_params(i, seed=42)
        records.append({
            "id": i,
            "filename": fname,
            "filepath": os.path.join(DATASET_DIR, fname),
            "subset": subset,
            "degradation": deg_params,
        })

    manifest = {
        "total_images": len(records),
        "train_opt_count": sum(1 for r in records if r["subset"] == "train_opt"),
        "test_live_count": sum(1 for r in records if r["subset"] == "test_live"),
        "image_size": target_size,
        "images": records,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDONE! Dataset upgraded: {manifest['total_images']} crisp images "
          f"({manifest['train_opt_count']} train_opt, {manifest['test_live_count']} test_live).")
    print("Now run 'python -m streamlit run app.py' to see crystal-clear images in your live demo!")


if __name__ == "__main__":
    download_and_build_highres(target_size=256)