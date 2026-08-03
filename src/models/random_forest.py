"""Random Forest classifier on Morgan fingerprints — classical ML baseline.

This model does NOT use the graph structure at all. It works from a
2048-bit Morgan fingerprint (ECFP4) of each molecule, which is the
classical way to represent molecules for ML before GNNs.

Why this matters as a baseline:
  - If GNNs can't beat a well-tuned RF, it suggests the graph structure
    isn't adding much beyond what fingerprints already capture.
  - A strong RF baseline is expected (~0.85–0.90 AUC on scaffold split).
  - Random Forest is robust to hyperparameters and rarely overfits badly.

Run this script:
    uv run python -m src.models.random_forest
"""

import json
import numpy as np
from pathlib import Path
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.features.morgan_fingerprints import build_fingerprint_dataset
from src.eval.metrics import evaluate_model, print_metrics, save_metrics

SEED = 42
RESULTS_DIR = Path("results/person_b")


def train_and_evaluate() -> tuple[RandomForestClassifier, dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load frozen dataset ──────────────────────────────────────────────
    print("Loading frozen scaffold-split dataset...")
    train_data = torch.load("data/processed/train.pt", weights_only=False)
    val_data   = torch.load("data/processed/val.pt",   weights_only=False)
    test_data  = torch.load("data/processed/test.pt",  weights_only=False)
    print(f"  Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

    # ── 2. Build Morgan fingerprints ────────────────────────────────────────
    # We extract SMILES from each PyG Data object and generate fingerprints.
    # This is fully independent of the GNN node features in .x.
    print("\nGenerating Morgan fingerprints (radius=2, 2048 bits)...")
    X_train, y_train = build_fingerprint_dataset(train_data)
    X_val,   y_val   = build_fingerprint_dataset(val_data)
    X_test,  y_test  = build_fingerprint_dataset(test_data)
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  Train label balance: {y_train.mean():.1%} BBB+")

    # ── 3. Train Random Forest ──────────────────────────────────────────────
    print("\nTraining Random Forest...")
    print("  n_estimators=500, max_features='sqrt', class_weight='balanced'")
    rf = RandomForestClassifier(
        n_estimators=500,
        # max_features='sqrt' — each tree sees sqrt(2048)≈45 features per split.
        # Reduces correlation between trees → better ensemble diversity.
        max_features="sqrt",
        # class_weight='balanced' — automatically adjusts weights inversely
        # proportional to class frequencies. Handles ~75/25 BBB+/BBB- imbalance.
        class_weight="balanced",
        # Use all available CPU cores. Set to 1 for reproducible timing.
        n_jobs=-1,
        random_state=SEED,
        # oob_score gives a free validation estimate from out-of-bag samples
        oob_score=True,
    )
    rf.fit(X_train, y_train)
    print(f"  Out-of-bag score (accuracy): {rf.oob_score_:.4f}")

    # ── 4. Evaluate on val and test ─────────────────────────────────────────
    all_results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        # predict_proba returns [P(BBB-), P(BBB+)] — we want column 1
        y_prob = rf.predict_proba(X)[:, 1]
        metrics = evaluate_model(y_true=y, y_prob=y_prob)
        print_metrics(metrics, model_name="RandomForest (Morgan ECFP4)", split=split_name)
        save_metrics(metrics, model_name="random_forest", split=split_name, out_dir=RESULTS_DIR)
        all_results[split_name] = metrics

    # ── 5. Save trained model ───────────────────────────────────────────────
    model_path = RESULTS_DIR / "random_forest_model.joblib"
    joblib.dump(rf, model_path)
    print(f"\nModel saved  → {model_path}")

    # ── 6. Feature importance (top 20 most important fingerprint bits) ──────
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    print(f"\nTop 20 most informative fingerprint bits:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. bit {idx:4d} — importance {importances[idx]:.5f}")

    return rf, all_results


if __name__ == "__main__":
    train_and_evaluate()
