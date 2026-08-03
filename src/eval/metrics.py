"""Shared evaluation script — single source of truth for all 5 models.

IMPORTANT: Both Person A and Person B import from this file.
Never calculate metrics anywhere else — use evaluate_model() here so
all model results are directly comparable.

Usage
-----
    from src.eval.metrics import evaluate_model, print_metrics, save_metrics

    y_true = np.array([0, 1, 1, 0, ...])   # ground truth labels
    y_prob = np.array([0.1, 0.9, 0.7, ...]) # predicted probabilities for BBB+

    metrics = evaluate_model(y_true, y_prob)
    print_metrics(metrics, model_name="GAT", split="test")
    save_metrics(metrics, model_name="gat", split="test", out_dir=Path("results/person_b"))
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute all evaluation metrics for a binary classifier.

    Args:
        y_true:    Ground truth binary labels, shape (N,). Values must be 0 or 1.
        y_prob:    Predicted probabilities for the positive (BBB+) class, shape (N,).
                   Values must be in [0, 1].
        threshold: Decision boundary for converting probabilities to hard predictions.
                   Default 0.5 is standard; tune on val set if needed.

    Returns:
        dict with keys:
            roc_auc   - Area under the ROC curve (threshold-independent)
            accuracy  - Fraction of correct hard predictions
            precision - Of predicted BBB+, how many are truly BBB+?
            recall    - Of truly BBB+, how many did we catch?
            f1        - Harmonic mean of precision and recall
            tn, fp, fn, tp - Raw confusion matrix counts
    """
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics(metrics: dict, model_name: str = "", split: str = "test") -> None:
    """Pretty-print a metrics dict to stdout."""
    header = f"{model_name} — {split.upper()}" if model_name else split.upper()
    print(f"\n{'='*52}")
    print(f"  {header}")
    print(f"{'='*52}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={metrics['tn']:4d}  FP={metrics['fp']:4d}")
    print(f"    FN={metrics['fn']:4d}  TP={metrics['tp']:4d}")


def save_metrics(metrics: dict, model_name: str, split: str, out_dir: Path) -> None:
    """Persist metrics to a JSON file.

    File is written to: out_dir/{model_name}_{split}.json

    Args:
        metrics:    dict returned by evaluate_model()
        model_name: short identifier, e.g. "gat", "gin", "random_forest"
        split:      "val" or "test"
        out_dir:    directory to save into (created if it doesn't exist)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{model_name}_{split}.json"
    payload = {
        "model": model_name,
        "split": split,
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Metrics saved → {out_path}")
