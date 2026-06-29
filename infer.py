"""
infer.py
========
Run inference on one image or a directory of images using the trained
cross-mark detection model. Prints per-image results and overall timing.

Usage:
    # Single image
    python infer.py --model runs/detect/lottery_cross_fp32_4gpu/weights/best.pt \
                    --source path/to/image.jpg

    # Directory  (saves annotated images to --output)
    python infer.py --model best.pt --source dataset/images/test/ \
                    --output predictions/ --conf 0.5

    # Show timing only (no saved output)
    python infer.py --model best.pt --source dataset/images/test/ --no-save
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── colour palette (BGR) for drawing boxes ────────────────────────────────────
BOX_COLOR    = (0, 200, 0)      # green
TEXT_COLOR   = (255, 255, 255)  # white
TEXT_BG      = (0, 200, 0)


def draw_boxes(image: np.ndarray, boxes, confs, class_names) -> np.ndarray:
    img = image.copy()
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"cross {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), TEXT_BG, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    return img


def run(
    model_path: str,
    source: str,
    conf: float = 0.25,
    iou: float = 0.5,
    output_dir: str | None = "predictions",
    device: str = "cpu",
    save: bool = True,
) -> dict:
    model = YOLO(model_path)
    source_path = Path(source)

    # Collect images
    if source_path.is_file():
        images = [source_path]
    else:
        images = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in source_path.glob(ext)
        )

    if not images:
        raise ValueError(f"No images found at {source}")

    out_dir = Path(output_dir) if (save and output_dir) else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    latencies_ms: list[float] = []

    print(f"Model  : {model_path}")
    print(f"Source : {source}  ({len(images)} images)")
    print(f"Conf   : {conf}  |  IoU: {iou}  |  Device: {device}")
    print("-" * 60)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        t0 = time.perf_counter()
        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        latencies_ms.append(latency_ms)

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        confs  = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        n_det = len(boxes)
        total_detections += n_det

        print(f"{img_path.name:<40} detections={n_det:3d}  latency={latency_ms:6.1f} ms")

        if save and out_dir:
            annotated = draw_boxes(frame, boxes, confs, model.names)
            cv2.imwrite(str(out_dir / img_path.name), annotated)

    # Summary
    n = len(latencies_ms)
    if n == 0:
        print("No images processed.")
        return {}

    p50  = float(np.percentile(latencies_ms, 50))
    p95  = float(np.percentile(latencies_ms, 95))
    p99  = float(np.percentile(latencies_ms, 99))
    mean = float(np.mean(latencies_ms))

    print("-" * 60)
    print(f"Images processed : {n}")
    print(f"Total detections : {total_detections}")
    print(f"Avg detections   : {total_detections / n:.1f} per image")
    print(f"Latency  mean    : {mean:.1f} ms")
    print(f"Latency  p50     : {p50:.1f} ms")
    print(f"Latency  p95     : {p95:.1f} ms")
    print(f"Latency  p99     : {p99:.1f} ms")
    if out_dir:
        print(f"Saved to         : {out_dir}/")

    return {
        "n_images":          n,
        "total_detections":  total_detections,
        "latency_mean_ms":   mean,
        "latency_p50_ms":    p50,
        "latency_p95_ms":    p95,
        "latency_p99_ms":    p99,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True,              help="Path to .pt weights")
    ap.add_argument("--source",  required=True,              help="Image file or directory")
    ap.add_argument("--conf",    type=float, default=0.25,   help="Confidence threshold")
    ap.add_argument("--iou",     type=float, default=0.5,    help="NMS IoU threshold")
    ap.add_argument("--output",  default="predictions",      help="Output directory for annotated images")
    ap.add_argument("--device",  default="cpu",              help="'cpu' or GPU index e.g. '0'")
    ap.add_argument("--no-save", action="store_true",        help="Skip saving annotated images")
    args = ap.parse_args()

    run(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        output_dir=args.output,
        device=args.device,
        save=not args.no_save,
    )
