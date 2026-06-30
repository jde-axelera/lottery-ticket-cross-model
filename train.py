"""
train.py
========
Train YOLOv8n for cross-mark detection on lottery tickets.
Logs every run to the local MLflow server at http://localhost:5000.

Usage:
    python train.py --data dataset/dataset.yaml --batch 32 --epochs 150
    python train.py --model yolov8s.pt --name lottery_cross_v2  # larger model
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

try:
    from mlflow_logger import MLflowLogger
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# Augmentation config logged to MLflow for reproducibility
AUGMENTATION_CONFIG = {
    "offline_aug.perspective":         "p=0.5, scale=(0.05,0.20)",
    "offline_aug.affine":              "p=0.4, translate=6%, scale=±15%, rotate=±6°",
    "offline_aug.grid_distortion":     "p=0.2",
    "offline_aug.blur":                "p=0.4 (Blur/MotionBlur/GaussianBlur)",
    "offline_aug.gauss_noise":         "p=0.5, std=(0.05,0.15)",
    "offline_aug.iso_noise":           "p=0.3",
    "offline_aug.downscale":           "p=0.4, scale=(0.30,0.75)",
    "offline_aug.jpeg_compression":    "p=0.5, quality=(30,85)",
    "offline_aug.brightness_contrast": "p=0.7, ±0.4",
    "offline_aug.random_gamma":        "p=0.4, (60,140)",
    "offline_aug.clahe":               "p=0.3",
    "offline_aug.hsv":                 "p=0.5",
    "offline_aug.random_shadow":       "p=0.3",
    "offline_aug.random_fog":          "p=0.1",
    "offline_aug.rgb_shift":           "p=0.3",
    "offline_aug.to_gray":             "p=0.1",
    "offline_aug.ink_colors":          "12 colors (dark blue/red/black/navy + 8 extras)",
    "offline_aug.crosses_per_subcol":  "1 or 2 (random)",
    "offline_aug.cross_size_range":    "(10-14, 18-25) pixels",
    "dataset.split":                   "70/20/10",
    "dataset.total_images":            "85000",
}


def train(
    data: str,
    batch: int = 128,       # 32 per GPU × 4 GPUs
    epochs: int = 150,
    imgsz: int = 640,
    model_name: str = 'yolov8n.pt',
    run_name: str = 'lottery_cross_v1',
    device: str = '0,1,2,3',
    use_mlflow: bool = True,
    mlflow_experiment: str = 'lottery-ticket-cross-model',
    lr0: float = 0.001,
    patience: int = 30,
) -> None:

    hyp = dict(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch,
        device=device,
        workers=8, optimizer='AdamW', lr0=lr0, lrf=0.01,
        warmup_epochs=3, patience=patience,
        amp=False,   # full float32 training
        mosaic=1.0, mixup=0.1, copy_paste=0.1,
        degrees=5.0, translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        conf=0.001, iou=0.7,
        project='/data/lottery-ticket-cross-model/runs/detect',
        name=run_name, save=True, plots=True, verbose=True,
    )
    params = {**hyp, 'model': model_name, **AUGMENTATION_CONFIG}
    tags = {'task': 'cross-detection', 'card': 'loto-libanais'}

    model = YOLO(model_name)

    if use_mlflow and _MLFLOW_AVAILABLE:
        logger = MLflowLogger(experiment=mlflow_experiment)
        model.add_callback("on_train_epoch_end", logger.on_epoch_end)
        model.add_callback("on_train_end",       logger.on_train_end)
        with logger.start_run(params=params, tags=tags, run_name=run_name):
            results = model.train(**hyp)
    else:
        if use_mlflow and not _MLFLOW_AVAILABLE:
            print("MLflow not available — training without logging.")
        results = model.train(**hyp)

    print(f"\nTraining complete. Best weights: {hyp['project']}/{run_name}/weights/best.pt")
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',        default='dataset/dataset.yaml')
    ap.add_argument('--batch',       type=int,   default=128)
    ap.add_argument('--epochs',      type=int,   default=150)
    ap.add_argument('--imgsz',       type=int,   default=640)
    ap.add_argument('--model',       default='yolov8n.pt')
    ap.add_argument('--name',        default='lottery_cross_v1')
    ap.add_argument('--device',      default='0,1,2,3')
    ap.add_argument('--no-mlflow',   action='store_true')
    ap.add_argument('--experiment',  default='lottery-ticket-cross-model')
    ap.add_argument('--lr0',         type=float, default=0.001,
                    help='Initial LR (use 0.0001 for fine-tuning from existing weights)')
    ap.add_argument('--patience',    type=int,   default=30)
    args = ap.parse_args()

    train(
        data=args.data,
        batch=args.batch,
        epochs=args.epochs,
        imgsz=args.imgsz,
        model_name=args.model,
        run_name=args.name,
        device=args.device,
        use_mlflow=not args.no_mlflow,
        mlflow_experiment=args.experiment,
        lr0=args.lr0,
        patience=args.patience,
    )
