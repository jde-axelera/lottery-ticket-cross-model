# Lottery Ticket Cross Model

Synthetic data generator and YOLO training pipeline for detecting hand-drawn cross marks on lottery card scans (Loto Libanais 1920×721 format).

## Overview

The generator renders randomised, handwriting-style cross marks directly onto a blank lottery card scan and produces YOLO-format bounding-box annotations. Every image is different — ink colour, cross style, size, tilt, and position are all randomised — so a model trained on this data never memorises a fixed layout.

Four cross styles are supported, all mapped to a single `cross` class (class 0):

| Style | Description |
|-------|-------------|
| X | Regular diagonal X with slight tilt jitter |
| thick_X | Heavy double-stroke X |
| + | Plus/cross with tilt jitter |
| double_X | Two overlapping X marks at a slight offset |

## Files

| File | Purpose |
|------|---------|
| `cross_generator.py` | Main library — dataset generator, disk writer, PyTorch `Dataset`, and CLI |
| `gen_crosses.py` | Quick script to render one sample image for visual inspection |
| `image.png` | Blank Loto Libanais card scan (base image for generation) |
| `image_marked.png` | Reference scan with human-drawn marks |
| `image_crosses.png` | Output of `gen_crosses.py` — example generated image |

## Usage

### Generate a YOLO dataset to disk

```bash
python cross_generator.py image.png ./dataset --samples 2000 --per-subcol 1
```

Output layout (Ultralytics-compatible):

```
dataset/
  dataset.yaml
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

### Use as a PyTorch Dataset

```python
from cross_generator import CrossDataset
from torch.utils.data import DataLoader

ds = CrossDataset('image.png', size=5000, crosses_per_subcol=1)
loader = DataLoader(ds, batch_size=8, num_workers=4,
                    collate_fn=lambda b: tuple(zip(*b)))

for imgs, labels in loader:
    # imgs: tuple of np.ndarray (H, W, 3) uint8
    # labels: tuple of lists of (class_id, cx, cy, w, h) normalised
    ...
```

### Infinite stream for custom training loops

```python
from cross_generator import stream

for img, labels in stream('image.png', crosses_per_subcol=2):
    train_step(img, labels)
```

### Python API

```python
from PIL import Image
from cross_generator import generate_sample

base = Image.open('image.png')
img_array, yolo_labels = generate_sample(base, crosses_per_subcol=1, size_range=(13, 20))
```

## Grid Geometry

The card has 8 networks, each with 5 sub-columns and 10 number rows. The generator places marks only at valid grid positions derived from the 1920×721 scan:

- 8 network X-starts: `[198, 360, 528, 699, 869, 1040, 1212, 1380]`
- 5 sub-column offsets per network: `[17, 50, 83, 116, 149]`
- Sub-column 0 (the "40s" column) has only 2 rows (rows 0–1)

## Requirements

```
Pillow
numpy
```

PyTorch and torchvision are optional — only needed when using `CrossDataset` with a `DataLoader`.

## Train with Ultralytics YOLOv8

```bash
python cross_generator.py image.png ./dataset --samples 5000
yolo detect train data=dataset/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```
