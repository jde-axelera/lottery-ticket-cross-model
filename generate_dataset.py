"""
generate_dataset.py
===================
Generate 85,000 augmented images for YOLOv8 cross-mark detection training.

Split: 70% train (59,500) / 20% val (17,000) / 10% test (8,500)
- Train: heavy augmentation simulating real lottery-shop cameras
- Val:   medium augmentation
- Test:  no augmentation (clean synthetic baseline for 100% P&R target)

Usage:
    python generate_dataset.py --output /data/lottery-ticket-cross-model/dataset \
                               --base-image /data/lottery-ticket-cross-model/image.png \
                               --total 85000 --workers 40
"""

from __future__ import annotations

import argparse
import random
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from cross_generator import generate_sample

# ---------------------------------------------------------------------------
# Extended ink palette
# ---------------------------------------------------------------------------
_EXTRA_COLORS = [
    (0,   80, 200, 240),
    (180,  0,  20, 230),
    (50,  50,  50, 220),
    (0,  120,  50, 220),
    (100,  0, 160, 230),
    (0,    0,   0, 255),
    (160, 40,   0, 220),
    (30,  30, 100, 240),
]

import cross_generator as _cg
_cg._INK_COLORS = _cg._INK_COLORS + _EXTRA_COLORS


# ---------------------------------------------------------------------------
# Albumentations pipelines
# ---------------------------------------------------------------------------

def _make_train_transform() -> A.Compose:
    return A.Compose([
        A.Perspective(scale=(0.05, 0.20), p=0.5),
        A.Affine(translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
                 scale=(0.85, 1.15), rotate=(-6, 6), p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.OneOf([
            A.Blur(blur_limit=7),
            A.MotionBlur(blur_limit=15),
            A.GaussianBlur(blur_limit=(3, 7)),
        ], p=0.4),
        A.GaussNoise(std_range=(0.05, 0.15), p=0.5),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.Downscale(scale_range=(0.3, 0.75), p=0.4),
        A.ImageCompression(quality_range=(30, 85), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
        A.RandomGamma(gamma_limit=(60, 140), p=0.4),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.5),
        A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=5, p=0.3),
        A.RandomFog(fog_coef_range=(0.05, 0.25), p=0.1),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.ToGray(p=0.1),
    ],
    bbox_params=A.BboxParams(
        format='yolo', label_fields=['class_labels'], min_visibility=0.3))


def _make_val_transform() -> A.Compose:
    return A.Compose([
        A.Blur(blur_limit=5, p=0.2),
        A.GaussNoise(std_range=(0.03, 0.10), p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.ImageCompression(quality_range=(50, 90), p=0.3),
    ],
    bbox_params=A.BboxParams(
        format='yolo', label_fields=['class_labels'], min_visibility=0.3))


# ---------------------------------------------------------------------------
# Worker function (runs in separate process)
# ---------------------------------------------------------------------------

_BASE_IMAGE_PATH: str = ''  # set by initializer
_BASE_IMAGE: Image.Image | None = None


def _init_worker(base_image_path: str) -> None:
    global _BASE_IMAGE, _BASE_IMAGE_PATH
    _BASE_IMAGE_PATH = base_image_path
    _BASE_IMAGE = Image.open(base_image_path)
    import cross_generator as cg
    cg._INK_COLORS = cg._INK_COLORS + _EXTRA_COLORS if len(cg._INK_COLORS) <= 4 else cg._INK_COLORS


def _generate_one(args: tuple) -> None:
    stem, split, out_dir = args
    out = Path(out_dir)

    if split == 'train':
        tf = _make_train_transform()
    elif split == 'val':
        tf = _make_val_transform()
    else:
        tf = None

    cpc = random.choice([1, 1, 2])
    size_lo = random.randint(10, 14)
    size_hi = random.randint(18, 25)

    img_arr, labels = generate_sample(
        _BASE_IMAGE,
        crosses_per_subcol=cpc,
        size_range=(size_lo, size_hi),
    )

    bboxes = [(lbl[1], lbl[2], lbl[3], lbl[4]) for lbl in labels]
    class_labels = [lbl[0] for lbl in labels]

    if tf is not None and bboxes:
        result = tf(image=img_arr, bboxes=bboxes, class_labels=class_labels)
        img_out = result['image']
        if img_out.ndim == 2:
            img_out = np.stack([img_out] * 3, axis=-1)
        out_bboxes = result['bboxes']
        out_classes = result['class_labels']
    else:
        img_out = img_arr
        out_bboxes = bboxes
        out_classes = class_labels

    cv2.imwrite(
        str(out / 'images' / split / f"{stem}.jpg"),
        cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    label_lines = '\n'.join(
        f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}"
        for cls, (bx, by, bw, bh) in zip(out_classes, out_bboxes)
    )
    (out / 'labels' / split / f"{stem}.txt").write_text(label_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(base_image_path: str, output_dir: str,
             total: int = 85_000, workers: int = 40) -> None:
    out = Path(output_dir)
    counts = {
        'train': int(total * 0.70),
        'val':   int(total * 0.20),
        'test':  total - int(total * 0.70) - int(total * 0.20),
    }

    print(f"Generating {total} images → {out}  (workers={workers})")
    for s, n in counts.items():
        print(f"  {s}: {n}")
    print()

    for s in counts:
        (out / 'images' / s).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / s).mkdir(parents=True, exist_ok=True)

    # Build full task list
    tasks = []
    idx = 0
    for split, n in counts.items():
        for _ in range(n):
            tasks.append((f"{idx:07d}", split, str(out)))
            idx += 1

    with Pool(workers, initializer=_init_worker,
              initargs=(base_image_path,)) as pool:
        for _ in tqdm(pool.imap_unordered(_generate_one, tasks, chunksize=20),
                      total=len(tasks), unit='img', desc='generating'):
            pass

    yaml_content = (
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: 1\n"
        f"names: ['cross']\n"
    )
    (out / 'dataset.yaml').write_text(yaml_content)
    print(f"\nDone. dataset.yaml written to {out}/dataset.yaml")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-image', default='image.png')
    ap.add_argument('--output', default='dataset')
    ap.add_argument('--total', type=int, default=85_000)
    ap.add_argument('--workers', type=int, default=min(40, cpu_count()))
    args = ap.parse_args()
    generate(args.base_image, args.output, args.total, args.workers)
