"""
train.py
========
Train YOLOv8n for cross-mark detection on lottery tickets.

Usage:
    python train.py --data dataset/dataset.yaml --batch 32 --epochs 150
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(data: str, batch: int = 32, epochs: int = 150,
          imgsz: int = 640, model_name: str = 'yolov8n.pt',
          run_name: str = 'lottery_cross_v1') -> None:

    model = YOLO(model_name)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=8,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        patience=30,
        # Built-in YOLOv8 augmentations (ON TOP of pre-augmented images)
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Confidence / NMS
        conf=0.001,
        iou=0.7,
        # Output
        project='/data/lottery-ticket-cross-model/runs/detect',
        name=run_name,
        save=True,
        plots=True,
        verbose=True,
    )

    print(f"\nTraining complete. Best weights: runs/detect/{run_name}/weights/best.pt")
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',    default='dataset/dataset.yaml')
    ap.add_argument('--batch',   type=int, default=32)
    ap.add_argument('--epochs',  type=int, default=150)
    ap.add_argument('--imgsz',   type=int, default=640)
    ap.add_argument('--model',   default='yolov8n.pt')
    ap.add_argument('--name',    default='lottery_cross_v1')
    args = ap.parse_args()

    train(
        data=args.data,
        batch=args.batch,
        epochs=args.epochs,
        imgsz=args.imgsz,
        model_name=args.model,
        run_name=args.name,
    )
