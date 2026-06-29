"""
mlflow_logger.py
================
MLflow integration for YOLOv8 training runs.

Registers per-run:
  - Git commit hash + branch
  - All training hyperparameters
  - Preprocessing / augmentation config
  - Per-epoch train/val metrics (loss, precision, recall, mAP)
  - Final test metrics
  - Training curve plots
  - Best model weights as artifact

Usage in train.py:
    from mlflow_logger import MLflowLogger
    logger = MLflowLogger(experiment="lottery-ticket-cross-model")
    model.add_callback("on_train_epoch_end", logger.on_epoch_end)
    model.add_callback("on_train_end",       logger.on_train_end)
    with logger.start_run(params, tags):
        model.train(...)
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

TRACKING_URI = "http://localhost:5001"


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_info() -> dict[str, str]:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                           cwd=Path(__file__).parent).decode().strip()
        except Exception:
            return "unknown"

    return {
        "git.commit":  _run(["git", "rev-parse", "--short", "HEAD"]),
        "git.branch":  _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git.message": _run(["git", "log", "-1", "--format=%s"]),
    }


# ---------------------------------------------------------------------------
# Callback class
# ---------------------------------------------------------------------------

class MLflowLogger:
    """
    Attach to a YOLO model via `model.add_callback(...)` and wrap
    `model.train(...)` with `logger.start_run(...)`.

    Parameters
    ----------
    experiment   : MLflow experiment name (created if absent).
    tracking_uri : MLflow server URL (default: http://localhost:5000).
    """

    def __init__(
        self,
        experiment: str = "default",
        tracking_uri: str = TRACKING_URI,
    ) -> None:
        self.experiment = experiment
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._run: mlflow.ActiveRun | None = None

    @contextmanager
    def start_run(
        self,
        params: dict[str, Any],
        tags: dict[str, str] | None = None,
        run_name: str | None = None,
    ):
        """Context manager that opens an MLflow run, logs params+tags, yields."""
        all_tags = _git_info()
        if tags:
            all_tags.update(tags)

        with mlflow.start_run(run_name=run_name, tags=all_tags) as run:
            self._run = run
            mlflow.log_params({k: str(v) for k, v in params.items()})
            print(f"MLflow run: {run.info.run_id}  "
                  f"({mlflow.get_tracking_uri()}/experiments/"
                  f"{run.info.experiment_id}/runs/{run.info.run_id})")
            try:
                yield run
            finally:
                self._run = None

    # ------------------------------------------------------------------
    # YOLOv8 callbacks
    # ------------------------------------------------------------------

    def on_epoch_end(self, trainer) -> None:
        """Log per-epoch metrics from the YOLOv8 trainer object."""
        if not mlflow.active_run():
            return
        metrics = {}
        if hasattr(trainer, 'loss'):
            metrics['train/box_loss']  = float(trainer.loss[0]) if len(trainer.loss) > 0 else None
            metrics['train/cls_loss']  = float(trainer.loss[1]) if len(trainer.loss) > 1 else None
            metrics['train/dfl_loss']  = float(trainer.loss[2]) if len(trainer.loss) > 2 else None
        if hasattr(trainer, 'metrics'):
            for k, v in trainer.metrics.items():
                try:
                    metrics[f'val/{k}'] = float(v)
                except (TypeError, ValueError):
                    pass
        if hasattr(trainer, 'lr'):
            for i, lr in enumerate(trainer.lr.values()):
                metrics[f'lr/pg{i}'] = float(lr)

        metrics = {k: v for k, v in metrics.items() if v is not None}
        epoch = getattr(trainer, 'epoch', 0)
        mlflow.log_metrics(metrics, step=epoch)

    def on_train_end(self, trainer) -> None:
        """Log final artifacts: best weights, training curves, confusion matrix."""
        if not mlflow.active_run():
            return

        save_dir = Path(trainer.save_dir) if hasattr(trainer, 'save_dir') else None
        if save_dir and save_dir.exists():
            # Best weights
            best = save_dir / 'weights' / 'best.pt'
            if best.exists():
                mlflow.log_artifact(str(best), artifact_path='weights')

            # Training curve plots
            for pattern in ['results.png', 'confusion_matrix.png',
                             'P_curve.png', 'R_curve.png', 'PR_curve.png',
                             'F1_curve.png']:
                for f in save_dir.glob(pattern):
                    mlflow.log_artifact(str(f), artifact_path='plots')

            # results CSV (all epoch metrics)
            results_csv = save_dir / 'results.csv'
            if results_csv.exists():
                mlflow.log_artifact(str(results_csv), artifact_path='plots')

        # Final val metrics
        if hasattr(trainer, 'metrics'):
            final = {f'final/{k}': float(v)
                     for k, v in trainer.metrics.items()
                     if isinstance(v, (int, float))}
            mlflow.log_metrics(final, step=getattr(trainer, 'epoch', 0))

    # ------------------------------------------------------------------
    # Standalone helper: log a completed evaluation result
    # ------------------------------------------------------------------

    def log_test_results(
        self,
        results,
        run_id: str | None = None,
        conf_threshold: float | None = None,
    ) -> None:
        """
        Log test-split evaluation results into an existing (or active) run.

        Parameters
        ----------
        results        : return value of model.val()
        run_id         : existing run ID to resume; uses active run if None
        conf_threshold : the confidence threshold used during evaluation
        """
        ctx = (mlflow.start_run(run_id=run_id)
               if run_id and not mlflow.active_run()
               else mlflow.active_run() or mlflow.start_run())

        with ctx:
            rd = results.results_dict
            test_metrics = {
                'test/precision': rd.get('metrics/precision(B)', 0),
                'test/recall':    rd.get('metrics/recall(B)', 0),
                'test/mAP50':     rd.get('metrics/mAP50(B)', 0),
                'test/mAP50_95':  rd.get('metrics/mAP50-95(B)', 0),
            }
            if conf_threshold is not None:
                test_metrics['test/conf_threshold'] = conf_threshold
            mlflow.log_metrics({k: float(v) for k, v in test_metrics.items()})
            print("Test metrics logged to MLflow:")
            for k, v in test_metrics.items():
                print(f"  {k}: {float(v):.4f}")
