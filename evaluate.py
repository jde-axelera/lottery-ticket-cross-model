"""
evaluate.py
===========
Evaluate a trained model on the test split and find the confidence threshold
that achieves 100% precision AND 100% recall simultaneously.

Usage:
    python evaluate.py --model runs/detect/lottery_cross_v1/weights/best.pt \
                       --data dataset/dataset.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

try:
    from mlflow_logger import MLflowLogger
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


def evaluate(model_path: str, data: str, iou: float = 0.5) -> None:
    model = YOLO(model_path)

    print("=" * 60)
    print("Running validation on TEST split (conf=0.001, full P-R curve)")
    print("=" * 60)

    # Use very low conf to get the full P-R curve
    results = model.val(
        data=data,
        split='test',
        conf=0.001,
        iou=iou,
        max_det=500,
        augment=False,
        verbose=True,
    )

    # Summary metrics
    mp   = float(results.results_dict.get('metrics/precision(B)', 0))
    mr   = float(results.results_dict.get('metrics/recall(B)', 0))
    map50 = float(results.results_dict.get('metrics/mAP50(B)', 0))
    map50_95 = float(results.results_dict.get('metrics/mAP50-95(B)', 0))

    print(f"\n{'=' * 60}")
    print(f"Test results (default threshold):")
    print(f"  Precision : {mp:.4f}")
    print(f"  Recall    : {mr:.4f}")
    print(f"  mAP@0.50  : {map50:.4f}")
    print(f"  mAP@.5:.95: {map50_95:.4f}")

    # Check P-R curve for a 100% P & R operating point
    try:
        p_curve = results.curves_results[0]  # precision curve
        r_curve = results.curves_results[1]  # recall curve
        # curves_results[0] shape: (nc, 1000) — one row per class, 1000 threshold points
        # corresponding x-axis is results.curves_results[4] (confidence thresholds)
        conf_axis = np.linspace(0, 1, 1000)

        p = np.array(p_curve[0])   # class 0
        r = np.array(r_curve[0])   # class 0

        # Find thresholds where both P >= 0.999 and R >= 0.999
        perfect_mask = (p >= 0.999) & (r >= 0.999)
        if perfect_mask.any():
            valid_confs = conf_axis[perfect_mask]
            best_conf = float(valid_confs.mean())
            print(f"\n✓ 100% P&R operating point found!")
            print(f"  Confidence threshold range: [{valid_confs.min():.3f}, {valid_confs.max():.3f}]")
            print(f"  Recommended threshold: {best_conf:.3f}")
        else:
            # Find the best combined F1 point
            f1 = 2 * p * r / (p + r + 1e-9)
            best_idx = int(np.argmax(f1))
            print(f"\n⚠ No single threshold gives 100% P&R.")
            print(f"  Best F1={f1[best_idx]:.4f} at conf={conf_axis[best_idx]:.3f}")
            print(f"  P={p[best_idx]:.4f}, R={r[best_idx]:.4f}")
            print("\n  → Consider: more training epochs, larger model (yolov8s), or TTA.")
    except Exception as e:
        print(f"\n(Could not parse P-R curves: {e})")

    # Also run with TTA
    print(f"\n{'=' * 60}")
    print("Running validation with Test-Time Augmentation (TTA)...")
    results_tta = model.val(
        data=data,
        split='test',
        conf=0.001,
        iou=iou,
        augment=True,
        verbose=False,
    )
    mp_tta   = float(results_tta.results_dict.get('metrics/precision(B)', 0))
    mr_tta   = float(results_tta.results_dict.get('metrics/recall(B)', 0))
    map50_tta = float(results_tta.results_dict.get('metrics/mAP50(B)', 0))
    print(f"TTA results:")
    print(f"  Precision : {mp_tta:.4f}")
    print(f"  Recall    : {mr_tta:.4f}")
    print(f"  mAP@0.50  : {map50_tta:.4f}")

    # Log test results to MLflow (looks up the most recent run for this model)
    if _MLFLOW_AVAILABLE:
        try:
            logger = MLflowLogger(experiment='lottery-ticket-cross-model')
            logger.log_test_results(results)
            logger.log_test_results(results_tta)
        except Exception as e:
            print(f"(MLflow logging skipped: {e})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data',  default='dataset/dataset.yaml')
    ap.add_argument('--iou',   type=float, default=0.5)
    args = ap.parse_args()
    evaluate(args.model, args.data, args.iou)
