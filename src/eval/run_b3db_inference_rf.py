"""Run inference on the deduplicated external B3DB set using the trained
Random Forest baseline (Morgan fingerprints).

Reuses the same dedup + graph-building logic as run_b3db_inference.py so
the Random Forest is evaluated on the exact same molecule set as GAT/GIN —
required for a fair 3-way external comparison.
"""

from pathlib import Path

import joblib

from src.eval.run_b3db_inference import load_and_dedup_b3db, build_b3db_graphs
from src.eval.metrics import evaluate_model, print_metrics, save_metrics
from src.features.morgan_fingerprints import build_fingerprint_dataset

RESULTS_DIR = Path("results/external/b3db")
RF_MODEL_PATH = Path("results/nethra/random_forest_model.joblib")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and deduplicating B3DB (same molecule set as GAT/GIN run)...")
    df = load_and_dedup_b3db()
    data_list = build_b3db_graphs(df)

    print(f"\nGenerating Morgan fingerprints for {len(data_list)} molecules...")
    X, y_true = build_fingerprint_dataset(data_list)
    print(f"  Fingerprint matrix: {X.shape}")
    print(f"  Label balance: {y_true.mean():.1%} BBB+")

    print(f"\nLoading Random Forest model from {RF_MODEL_PATH}...")
    rf = joblib.load(RF_MODEL_PATH)

    y_prob = rf.predict_proba(X)[:, 1]
    metrics = evaluate_model(y_true=y_true, y_prob=y_prob)

    print_metrics(metrics, model_name="RandomForest (external B3DB)", split="external")
    save_metrics(metrics, model_name="random_forest_b3db_external", split="external", out_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()