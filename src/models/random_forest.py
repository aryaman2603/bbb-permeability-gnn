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

    print("Loading frozen scaffold-split dataset...")
    train_data = torch.load("data/processed/train.pt", weights_only=False)
    val_data   = torch.load("data/processed/val.pt",   weights_only=False)
    test_data  = torch.load("data/processed/test.pt",  weights_only=False)
    print(f"  Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")


    print("\nGenerating Morgan fingerprints (radius=2, 2048 bits)...")
    X_train, y_train = build_fingerprint_dataset(train_data)
    X_val,   y_val   = build_fingerprint_dataset(val_data)
    X_test,  y_test  = build_fingerprint_dataset(test_data)
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  Train label balance: {y_train.mean():.1%} BBB+")

    print("\nTraining Random Forest...")
    print("  n_estimators=500, max_features='sqrt', class_weight='balanced'")
    rf = RandomForestClassifier(
        n_estimators=500,
     
        max_features="sqrt",
      
        class_weight="balanced",
      
        n_jobs=-1,
        random_state=SEED,
     
        oob_score=True,
    )
    rf.fit(X_train, y_train)
    print(f"  Out-of-bag score (accuracy): {rf.oob_score_:.4f}")


    all_results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_prob = rf.predict_proba(X)[:, 1]
        metrics = evaluate_model(y_true=y, y_prob=y_prob)
        print_metrics(metrics, model_name="RandomForest (Morgan ECFP4)", split=split_name)
        save_metrics(metrics, model_name="random_forest", split=split_name, out_dir=RESULTS_DIR)
        all_results[split_name] = metrics

    model_path = RESULTS_DIR / "random_forest_model.joblib"
    joblib.dump(rf, model_path)
    print(f"\nModel saved  → {model_path}")

    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    print(f"\nTop 20 most informative fingerprint bits:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. bit {idx:4d} — importance {importances[idx]:.5f}")

    return rf, all_results


if __name__ == "__main__":
    train_and_evaluate()
