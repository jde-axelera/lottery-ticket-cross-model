# Model Weights

**best.pt** — YOLOv8n trained for cross-mark detection

| Metric | Value |
|--------|-------|
| Precision | 1.0000 |
| Recall | 0.9551 |
| mAP@50 | 0.9550 |
| mAP@50-95 | 0.8049 |
| Training epochs | 150 |
| Best F1 conf threshold | 1.000 |

Available in the MLflow Model Registry as `lottery-cross-detector` version 1 (stage: Production).

Download via MLflow:
```python
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')
model_uri = 'models:/lottery-cross-detector/Production'
# Load with ultralytics:
from ultralytics import YOLO
YOLO(mlflow.artifacts.download_artifacts(model_uri))
```
