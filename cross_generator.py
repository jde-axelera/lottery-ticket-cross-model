"""
cross_generator.py
==================
On-the-fly YOLO data generator for cross-mark detection on lottery card images.

Classes
-------
CrossDataset   -- torch.utils.data.Dataset-compatible lazy generator
                  (generates a fresh annotated image on every __getitem__ call)

Functions
---------
generate_sample(base_image, ...)  -> (np.ndarray, list[YoloLabel])
    Draw random crosses on a copy of the base image; return image + YOLO labels.

write_dataset(base_image_path, output_dir, ...)
    Materialise N samples to disk in Ultralytics YOLO directory layout
    (images/train|val|test  +  labels/train|val|test  +  dataset.yaml).

stream(base_image_path, ...)  -> Iterator[(np.ndarray, list[YoloLabel])]
    Infinite generator — useful for custom training loops.

Cross classes
-------------
  0  X          regular diagonal X
  1  thick_X    heavy / double-stroke X
  2  plus        + cross
  3  double_X   two overlapping X's at a slight offset

YOLO label format (per mark)
----------------------------
  (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
  all coordinates normalised to [0, 1] relative to image size.

CLI usage
---------
  python cross_generator.py image.png ./dataset --samples 2000 --per-subcol 1

Typical PyTorch usage
---------------------
  from cross_generator import CrossDataset
  from torch.utils.data import DataLoader

  ds     = CrossDataset('image.png', size=5000)
  loader = DataLoader(ds, batch_size=8, num_workers=4,
                      collate_fn=lambda b: tuple(zip(*b)))
  for imgs, labels in loader:
      ...   # imgs: tuple of np.ndarray (H,W,3); labels: tuple of lists
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Cross class IDs
# ---------------------------------------------------------------------------
CLASS_X        = 0
CLASS_THICK_X  = 0   # same class — all marks are "cross"
CLASS_PLUS     = 0
CLASS_DOUBLE_X = 0

CROSS_CLASSES = [CLASS_X, CLASS_THICK_X, CLASS_PLUS, CLASS_DOUBLE_X]
CLASS_NAMES   = ["cross"]

# Type alias for a single YOLO annotation row
YoloLabel = tuple[int, float, float, float, float]  # cls cx cy w h  (all norm)

# ---------------------------------------------------------------------------
# Grid geometry — derived from 1920×721 Loto Libanais card scan
# ---------------------------------------------------------------------------
# x pixel start of each of the 8 network columns (left→right = net 8→1)
_NET_X_STARTS   = [198, 360, 528, 699, 869, 1040, 1212, 1380]
# x pixel offset of each sub-column within a network
_SUBCOL_OFFSETS = [17, 50, 83, 116, 149]
# y pixel centre of each of the 10 number rows
_ROW_YS         = [int(108 + i * 57.3) for i in range(10)]
# sub-column 0 (the "40s" column) only contains rows 0 and 1
_SUBCOL_MAX_ROW = [1, 9, 9, 9, 9]

# ---------------------------------------------------------------------------
# Ink colours  (RGBA)
# ---------------------------------------------------------------------------
_INK_COLORS = [
    (15,  15,  90, 230),   # dark blue
    (100,  0,   0, 220),   # dark red
    (10,  10,  10, 240),   # near-black
    (20,  60, 120, 220),   # blue-black
]

# ---------------------------------------------------------------------------
# Low-level drawing primitives
# ---------------------------------------------------------------------------

def _j(r: int = 4) -> int:
    return random.randint(-r, r)

def _rot(cx: float, cy: float, x: float, y: float, deg: float):
    a = math.radians(deg)
    dx, dy = x - cx, y - cy
    return cx + dx * math.cos(a) - dy * math.sin(a), \
           cy + dx * math.sin(a) + dy * math.cos(a)

def _seg(draw: ImageDraw.ImageDraw,
         x1: float, y1: float, x2: float, y2: float,
         color, width: int):
    """Wobbly two-segment line to simulate hand tremor."""
    mx, my = (x1 + x2) / 2 + _j(3), (y1 + y2) / 2 + _j(3)
    draw.line([(x1, y1), (mx, my)], fill=color, width=width)
    draw.line([(mx, my), (x2, y2)], fill=color, width=width)

def _draw_x(draw: ImageDraw.ImageDraw,
            cx: float, cy: float, size: int, color, width: int = 3):
    tilt = random.uniform(-8, 8)
    p = [_rot(0, 0, *c, tilt) for c in [(-size,-size),(size,size),(size,-size),(-size,size)]]
    _seg(draw, cx+p[0][0]+_j(3), cy+p[0][1]+_j(3),
               cx+p[1][0]+_j(3), cy+p[1][1]+_j(3), color, width)
    _seg(draw, cx+p[2][0]+_j(3), cy+p[2][1]+_j(3),
               cx+p[3][0]+_j(3), cy+p[3][1]+_j(3), color, width)

def _draw_thick_x(draw: ImageDraw.ImageDraw,
                  cx: float, cy: float, size: int, color):
    _draw_x(draw, cx, cy, size, color, width=4)
    _draw_x(draw, cx + _j(2), cy + _j(2), size, color, width=2)

def _draw_plus(draw: ImageDraw.ImageDraw,
               cx: float, cy: float, size: int, color, width: int = 3):
    tilt = random.uniform(-8, 8)
    for dx1, dy1, dx2, dy2 in [(-size, 0, size, 0), (0, -size, 0, size)]:
        p1 = _rot(cx, cy, cx + dx1, cy + dy1, tilt)
        p2 = _rot(cx, cy, cx + dx2, cy + dy2, tilt)
        _seg(draw, p1[0]+_j(2), p1[1]+_j(2), p2[0]+_j(2), p2[1]+_j(2), color, width)

def _draw_double_x(draw: ImageDraw.ImageDraw,
                   cx: float, cy: float, size: int, color):
    _draw_x(draw, cx,       cy,       size,     color, width=2)
    _draw_x(draw, cx+_j(3), cy+_j(3), size - 2, color, width=2)

_DRAW_FN = {
    CLASS_X:        _draw_x,
    CLASS_THICK_X:  _draw_thick_x,
    CLASS_PLUS:     _draw_plus,
    CLASS_DOUBLE_X: _draw_double_x,
}

# ---------------------------------------------------------------------------
# Public API — sample generator
# ---------------------------------------------------------------------------

def generate_sample(
    base_image: Image.Image,
    *,
    crosses_per_subcol: int = 1,
    size_range: tuple[int, int] = (13, 20),
    seed: int | None = None,
) -> tuple[np.ndarray, list[YoloLabel]]:
    """
    Draw handwritten-style crosses on a copy of *base_image*.

    Parameters
    ----------
    base_image        : Blank lottery card PIL Image (loaded once by caller).
    crosses_per_subcol: Number of crosses to place per sub-column (default 1).
                        Increase for denser annotation examples.
    size_range        : (min, max) half-size of each cross mark in pixels.
    seed              : Fix RNG seed for a reproducible sample.

    Returns
    -------
    image  : np.ndarray  shape (H, W, 3)  uint8
    labels : list of YoloLabel — (class_id, cx_norm, cy_norm, w_norm, h_norm)
    """
    if seed is not None:
        random.seed(seed)

    img  = base_image.copy().convert('RGBA')
    draw = ImageDraw.Draw(img)
    W, H = img.size

    labels: list[YoloLabel] = []
    color_idx = 0

    for ni, nx in enumerate(_NET_X_STARTS):
        for sci, sco in enumerate(_SUBCOL_OFFSETS):
            cx_base  = nx + sco
            max_row  = _SUBCOL_MAX_ROW[sci]

            for _ in range(crosses_per_subcol):
                cx   = cx_base + _j(3)
                cy   = _ROW_YS[random.randint(0, max_row)] + _j(3)
                size = random.randint(*size_range)
                cls  = random.choice(CROSS_CLASSES)
                col  = _INK_COLORS[color_idx % len(_INK_COLORS)]

                _DRAW_FN[cls](draw, cx, cy, size, col)
                color_idx += 1

                # Bounding box — add stroke margin so the bbox encloses the mark
                margin  = size + 6
                labels.append((
                    cls,
                    cx / W,
                    cy / H,
                    (margin * 2) / W,
                    (margin * 2) / H,
                ))

    return np.array(img.convert('RGB'), dtype=np.uint8), labels


# ---------------------------------------------------------------------------
# Public API — disk dataset writer
# ---------------------------------------------------------------------------

def write_dataset(
    base_image_path: str | Path,
    output_dir: str | Path,
    n_samples: int = 1000,
    *,
    crosses_per_subcol: int = 1,
    size_range: tuple[int, int] = (13, 20),
    split: tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> None:
    """
    Generate *n_samples* annotated images and save in Ultralytics YOLO layout.

    Output structure::

        output_dir/
          dataset.yaml
          images/train/  images/val/  images/test/
          labels/train/  labels/val/  labels/test/

    Parameters
    ----------
    base_image_path   : Path to the blank lottery card scan.
    output_dir        : Root directory for the generated dataset.
    n_samples         : Total number of images to generate.
    crosses_per_subcol: Crosses per sub-column per image.
    size_range        : (min, max) cross half-size in pixels.
    split             : (train, val, test) fractions — must sum to 1.
    """
    base = Image.open(base_image_path)
    out  = Path(output_dir)
    splits = ['train', 'val', 'test']
    thresh = [split[0], split[0] + split[1]]

    for s in splits:
        (out / 'images' / s).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / s).mkdir(parents=True, exist_ok=True)

    for i in range(n_samples):
        r    = i / n_samples
        fold = splits[0] if r < thresh[0] else (splits[1] if r < thresh[1] else splits[2])
        stem = f"{i:06d}"

        img_arr, labels = generate_sample(
            base,
            crosses_per_subcol=crosses_per_subcol,
            size_range=size_range,
        )
        Image.fromarray(img_arr).save(out / 'images' / fold / f"{stem}.png")
        label_txt = '\n'.join(
            f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}"
            for cls, bx, by, bw, bh in labels
        )
        (out / 'labels' / fold / f"{stem}.txt").write_text(label_txt)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples}")

    (out / 'dataset.yaml').write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(set(CROSS_CLASSES))}\n"
        f"names: {CLASS_NAMES}\n"
    )
    print(f"Dataset written → {out}  ({n_samples} samples)")


# ---------------------------------------------------------------------------
# Public API — infinite stream
# ---------------------------------------------------------------------------

def stream(
    base_image_path: str | Path,
    *,
    crosses_per_subcol: int = 1,
    size_range: tuple[int, int] = (13, 20),
) -> Iterator[tuple[np.ndarray, list[YoloLabel]]]:
    """
    Infinite generator of (image, labels) pairs.  Useful for custom loops::

        for img, labels in stream('image.png'):
            train_step(img, labels)
    """
    base = Image.open(base_image_path)
    while True:
        yield generate_sample(base, crosses_per_subcol=crosses_per_subcol,
                               size_range=size_range)


# ---------------------------------------------------------------------------
# Public API — PyTorch-compatible Dataset
# ---------------------------------------------------------------------------

class CrossDataset:
    """
    On-the-fly cross-mark detection dataset compatible with
    ``torch.utils.data.Dataset``.

    A fresh annotated image is generated on every ``__getitem__`` call,
    so every epoch sees different augmentation without any disk I/O.

    Parameters
    ----------
    base_image_path   : Path to the blank lottery card scan.
    size              : Virtual dataset length reported to the DataLoader.
    crosses_per_subcol: Crosses drawn per sub-column per image.
    size_range        : (min, max) cross half-size in pixels.
    transform         : Optional callable ``(np.ndarray) -> any`` applied to
                        the image before returning (e.g. torchvision transforms).

    Example
    -------
    >>> from torch.utils.data import DataLoader
    >>> from cross_generator import CrossDataset
    >>>
    >>> ds = CrossDataset('image.png', size=5000, crosses_per_subcol=2)
    >>> loader = DataLoader(ds, batch_size=8, num_workers=4,
    ...                     collate_fn=lambda b: tuple(zip(*b)))
    >>> for imgs, batch_labels in loader:
    ...     pass   # imgs: tuple of np.ndarray; batch_labels: tuple of lists
    """

    def __init__(
        self,
        base_image_path: str | Path,
        size: int = 1000,
        *,
        crosses_per_subcol: int = 1,
        size_range: tuple[int, int] = (13, 20),
        transform: Callable | None = None,
    ):
        self.base_image         = Image.open(base_image_path)
        self._size              = size
        self.crosses_per_subcol = crosses_per_subcol
        self.size_range         = size_range
        self.transform          = transform

    # -- Dataset protocol ------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> tuple[any, list[YoloLabel]]:
        img, labels = generate_sample(
            self.base_image,
            crosses_per_subcol=self.crosses_per_subcol,
            size_range=self.size_range,
        )
        if self.transform is not None:
            img = self.transform(img)
        return img, labels

    def __iter__(self) -> Iterator:
        for i in range(self._size):
            yield self[i]

    def __repr__(self) -> str:
        return (f"CrossDataset(size={self._size}, "
                f"crosses_per_subcol={self.crosses_per_subcol}, "
                f"size_range={self.size_range})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(
        description='Generate a YOLO cross-mark dataset from a lottery card scan.')
    ap.add_argument('base_image',    help='Blank lottery card image  (e.g. image.png)')
    ap.add_argument('output_dir',    help='Root directory for the generated dataset')
    ap.add_argument('--samples',     type=int, default=1000, metavar='N',
                    help='Total number of images to generate  (default: 1000)')
    ap.add_argument('--per-subcol',  type=int, default=1,    metavar='K',
                    help='Crosses per sub-column per image  (default: 1)')
    ap.add_argument('--min-size',    type=int, default=13,   metavar='PX')
    ap.add_argument('--max-size',    type=int, default=20,   metavar='PX')
    args = ap.parse_args()

    write_dataset(
        args.base_image,
        args.output_dir,
        n_samples=args.samples,
        crosses_per_subcol=args.per_subcol,
        size_range=(args.min_size, args.max_size),
    )
