"""
WEX 428 - Computer Vision Project
Module 1: Image Input and Dataset Management Module

Handles dataset preparation (minimum 2,500 images), train/test splitting,
and assigning deterministic per-image random degradation parameters so each
image has its own unique corruption setting.
"""

import os
import json
import random
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "images")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "dataset_manifest.json")
MIN_IMAGES = 2500
TRAIN_RATIO = 0.80  # 2,000 train/optimization images, 500 reserved test images


def _generate_random_degradation_params(img_id, seed):
    """Assign random degradation parameters specific to a single image."""
    rng = random.Random(seed + img_id * 17)
    task_type = rng.choice(["denoising", "deblurring"])
    noise_type = rng.choice(["gaussian", "salt_and_pepper"])
    gaussian_sigma = rng.choice([15.0, 25.0, 35.0, 50.0])
    sp_amount = rng.choice([0.03, 0.05, 0.10, 0.15])

    blur_type = rng.choice(["gaussian", "motion"])
    blur_kernel_size = rng.choice([5, 7, 9, 11])
    blur_sigma = round(rng.uniform(1.0, 2.5), 2)
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


def _create_procedural_natural_image(idx, size=128):
    """Fallback generator if offline: creates diverse multi-shape/texture test images."""
    rng = np.random.RandomState(idx + 1000)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xv, yv = np.meshgrid(x, y)

    c1 = rng.uniform(0.1, 0.9, size=(1, 1, 3))
    c2 = rng.uniform(0.1, 0.9, size=(1, 1, 3))
    angle = rng.uniform(0, np.pi)
    grad = (np.cos(angle) * xv + np.sin(angle) * yv)[..., None]
    img = c1 * grad + c2 * (1.0 - grad)

    num_shapes = rng.randint(3, 8)
    for _ in range(num_shapes):
        shape_type = rng.choice(["circle", "rect", "wave"])
        color = rng.uniform(0.05, 0.95, size=(3,))
        if shape_type == "circle":
            cx, cy = rng.uniform(0.15, 0.85, size=2)
            r = rng.uniform(0.08, 0.28)
            mask = (xv - cx) ** 2 + (yv - cy) ** 2 <= r ** 2
            img[mask] = color
        elif shape_type == "rect":
            x0, x1 = sorted(rng.uniform(0.1, 0.9, size=2))
            y0, y1 = sorted(rng.uniform(0.1, 0.9, size=2))
            mask = (xv >= x0) & (xv <= x1) & (yv >= y0) & (yv <= y1)
            img[mask] = color
        else:
            freq = rng.uniform(4.0, 14.0)
            wave = 0.12 * np.sin(2 * np.pi * freq * xv)[..., None]
            img = np.clip(img + wave, 0.0, 1.0)

    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_uint8)


def prepare_dataset(total_images=MIN_IMAGES, img_size=128, base_seed=42):
    """
    Ensures the dataset directory has at least total_images (default 2,500) images
    and builds the train/optimization and held-out test split manifest.
    First tries downloading real images via torchvision (CIFAR-10) if available.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if len(manifest.get("images", [])) >= total_images:
            return manifest

    print(f"Preparing dataset with {total_images} images in '{DATASET_DIR}'...")
    existing_files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    tracked_files = list(existing_files)

    if len(tracked_files) < total_images:
        saved_count = len(tracked_files)
        try:
            import torchvision
            raw_cache = os.path.join(BASE_DIR, "data", "torchvision_raw")
            os.makedirs(raw_cache, exist_ok=True)
            print("Downloading CIFAR-10 dataset via torchvision to extract 2,500+ diverse images...")
            ds = torchvision.datasets.CIFAR10(root=raw_cache, train=True, download=True)
            indices = list(range(len(ds)))
            rng = random.Random(base_seed)
            rng.shuffle(indices)
            for idx in indices:
                if saved_count >= total_images:
                    break
                pil_img, label = ds[idx]
                pil_img = pil_img.resize((img_size, img_size), Image.BICUBIC)
                fname = f"img_{saved_count:04d}_class{label}.png"
                pil_img.save(os.path.join(DATASET_DIR, fname))
                tracked_files.append(fname)
                saved_count += 1
        except Exception as e:
            print(f"Note: torchvision download skipped ({e}). Generating procedural test images instead.")

        while saved_count < total_images:
            pil_img = _create_procedural_natural_image(saved_count, size=img_size)
            fname = f"img_{saved_count:04d}.png"
            pil_img.save(os.path.join(DATASET_DIR, fname))
            tracked_files.append(fname)
            saved_count += 1

    all_files = sorted(list(set(tracked_files)))[:total_images]

    rng = random.Random(base_seed)
    shuffled_indices = list(range(len(all_files)))
    rng.shuffle(shuffled_indices)

    num_train = int(len(all_files) * TRAIN_RATIO)
    train_set_ids = set(shuffled_indices[:num_train])

    records = []
    for i, fname in enumerate(all_files):
        subset = "train_opt" if i in train_set_ids else "test_live"
        deg_params = _generate_random_degradation_params(i, base_seed)
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
        "image_size": img_size,
        "images": records,
    }

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset ready: {manifest['total_images']} total "
          f"({manifest['train_opt_count']} train_opt, {manifest['test_live_count']} reserved test_live).")
    return manifest


def get_subset_records(subset="test_live"):
    """Returns metadata records for either 'train_opt' or 'test_live'."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = prepare_dataset()
    return [r for r in manifest["images"] if r["subset"] == subset]


def load_image_rgb(filepath, target_size=(128, 128)):
    """Loads an image as a float32 RGB array normalized to [0.0, 1.0]."""
    img = Image.open(filepath).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)
