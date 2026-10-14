"""Step 2+3: tune RF-gated GIN cascade on val (accuracy-first), evaluate on test.

Protocol (no leakage):
  1. Score RF + GIN on frozen val/test splits.
  2. Grid-search (tau_low, tau_high) on VAL only, maximizing F1 (tie-break ROC-AUC).
  3. Fit soft-fusion LogisticRegression on VAL only (features [rf_prob, gin_prob]).
  4. Report best hard + soft gates on TEST vs RF-only / GIN-only, incl. routing %,
     RF-FP-fixed / RF-FN-fixed rates, and latency.

Usage:
  uv run python -m src.eval.evaluate_cascade
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import DataLoader

from src.eval.metrics import evaluate_model, print_metrics
from src.explain.aggregate_explanations import load_model
from src.features.morgan_fingerprints import build_fingerprint_dataset

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RF_PATH = "results/nethra/random_forest_model.joblib"
GIN_CKPT = "results/nethra/gin_best.pt"
OUT_DIR = Path("results/hybrid")

TAU_LOWS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
TAU_HIGHS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def rf_probs_for(data_list: list, rf) -> np.ndarray:
    X, _ = build_fingerprint_dataset(data_list)
    return rf.predict_proba(X)[:, 1]


def gin_probs_for(data_list: list, gin, batch_size: int = 64) -> np.ndarray:
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    out: list[np.ndarray] = []
    gin.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = gin(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def cascade_probs(rf_p: np.ndarray, gin_p: np.ndarray, lo: float, hi: float,
                  fusion=None) -> tuple[np.ndarray, np.ndarray]:
    routed = (rf_p >= lo) & (rf_p <= hi)
    if fusion is None:
        final = np.where(routed, gin_p, rf_p)
    elif not routed.any():
        final = rf_p.copy()
    else:
        feats = np.column_stack([rf_p[routed], gin_p[routed]])
        fused = fusion.predict_proba(feats)[:, 1]
        final = rf_p.copy()
        final[routed] = fused
    return final, routed


def main() -> None:
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models + frozen splits...")
    rf = joblib.load(RF_PATH)
    gin = load_model("gin", GIN_CKPT).to(DEVICE).eval()
    val = torch.load("data/processed/val.pt", weights_only=False)
    test = torch.load("data/processed/test.pt", weights_only=False)
    print(f"  Val: {len(val)}  Test: {len(test)}")

    print("\nScoring RF (fingerprints) + GIN (graphs)...")
    y_val = np.array([int(d.y.item()) for d in val])
    y_test = np.array([int(d.y.item()) for d in test])
    rf_val, rf_test = rf_probs_for(val, rf), rf_probs_for(test, rf)
    gin_val, gin_test = gin_probs_for(val, gin), gin_probs_for(test, gin)

    base = {
        "rf_val": evaluate_model(y_val, rf_val),
        "gin_val": evaluate_model(y_val, gin_val),
        "rf_test": evaluate_model(y_test, rf_test),
        "gin_test": evaluate_model(y_test, gin_test),
    }
    print("\nBaselines:")
    print_metrics(base["rf_val"], "RF", "val")
    print_metrics(base["gin_val"], "GIN", "val")
    print_metrics(base["rf_test"], "RF", "test")
    print_metrics(base["gin_test"], "GIN", "test")

    # --- Grid search on VAL, hard-switch ---
    print("\nGrid-searching gate on VAL (max F1, tie-break AUC)...")
    rows = []
    for lo, hi in product(TAU_LOWS, TAU_HIGHS):
        if lo >= hi:
            continue
        final, routed = cascade_probs(rf_val, gin_val, lo, hi)
        m = evaluate_model(y_val, final)
        rows.append({"tau_low": lo, "tau_high": hi,
                     "route_frac": float(routed.mean()),
                     "f1": m["f1"], "roc_auc": m["roc_auc"], "accuracy": m["accuracy"]})
    rows.sort(key=lambda r: (r["f1"], r["roc_auc"]), reverse=True)
    best = rows[0]
    print(f'  Best hard gate on val: lo={best["tau_low"]} hi={best["tau_high"]} '
          f'route={best["route_frac"]:.1%} F1={best["f1"]:.4f} AUC={best["roc_auc"]:.4f}')
    print("  Top 5 gates:")
    for r in rows[:5]:
        print(f'    lo={r["tau_low"]:.2f} hi={r["tau_high"]:.2f} '
              f'route={r["route_frac"]:.1%} F1={r["f1"]:.4f} AUC={r["roc_auc"]:.4f}')

    # --- Soft fusion fit on VAL only ---
    print("\nFitting soft-fusion LogisticRegression on VAL [rf_prob, gin_prob]...")
    fusion = LogisticRegression(max_iter=1000)
    fusion.fit(np.column_stack([rf_val, gin_val]), y_val)
    print(f"  coef={fusion.coef_.ravel().round(4).tolist()} intercept={fusion.intercept_.round(4).tolist()}")

    # --- TEST evaluation ---
    def slice_fixes(y, rf_p, gin_p, final_p, thr=0.5):
        rf_pred = (rf_p >= thr).astype(int)
        fin_pred = (final_p >= thr).astype(int)
        rf_fp = (rf_pred == 1) & (y == 0)
        rf_fn = (rf_pred == 0) & (y == 1)
        fixed_fp = int((((fin_pred == 0) & rf_fp)).sum())
        fixed_fn = int((((fin_pred == 1) & rf_fn)).sum())
        return {
            "rf_fp": int(rf_fp.sum()), "rf_fn": int(rf_fn.sum()),
            "fixed_fp": fixed_fp, "fixed_fn": fixed_fn,
            "fp_fix_rate": fixed_fp / max(1, int(rf_fp.sum())),
            "fn_fix_rate": fixed_fn / max(1, int(rf_fn.sum())),
        }

    results = {"baselines": base, "val_search": rows, "best_gate": best}
    for name, kw in [("hard", {}),
                     ("soft", {"fusion": fusion})]:
        final, routed = cascade_probs(rf_test, gin_test,
                                      best["tau_low"], best["tau_high"], **kw)
        m = evaluate_model(y_test, final)
        print_metrics(m, f'Cascade-{name} lo={best["tau_low"]} hi={best["tau_high"]}', "test")
        print(f'  Routed to GIN: {routed.mean():.1%} ({int(routed.sum())}/{len(routed)})')
        fixes = slice_fixes(y_test, rf_test, gin_test, final)
        print(f'  RF errors fixed: FP {fixes["fixed_fp"]}/{fixes["rf_fp"]} '
              f'({fixes["fp_fix_rate"]:.1%}), FN {fixes["fixed_fn"]}/{fixes["rf_fn"]} '
              f'({fixes["fn_fix_rate"]:.1%})')
        results[f"test_{name}"] = {"metrics": m, "route_frac": float(routed.mean()),
                                   "n_routed": int(routed.sum()), **fixes}

    # --- Latency (single-threaded, test set, includes featurization) ---
    print("\nLatency benchmark on test set...")
    t0 = time.perf_counter()
    _ = rf_probs_for(test, rf)
    t_rf = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = gin_probs_for(test, gin)
    t_gin = time.perf_counter() - t0
    # cascade: RF always + GIN on routed subset only
    lo, hi = best["tau_low"], best["tau_high"]
    routed_mask = (rf_test >= lo) & (rf_test <= hi)
    t0 = time.perf_counter()
    _ = gin_probs_for([d for d, r in zip(test, routed_mask) if r], gin)
    t_gin_routed = time.perf_counter() - t0
    lat = {"rf_s": t_rf, "gin_full_s": t_gin,
           "gin_routed_s": t_gin_routed,
           "cascade_est_s": t_rf + t_gin_routed,
           "route_frac": float(routed_mask.mean())}
    print(f'  RF-only: {t_rf:.2f}s | GIN-full: {t_gin:.2f}s | '
          f'Cascade-est: {lat["cascade_est_s"]:.2f}s (route {lat["route_frac"]:.1%})')
    results["latency"] = lat

    fusion_path = OUT_DIR / "fusion_logreg.joblib"
    joblib.dump(fusion, fusion_path)
    with open(OUT_DIR / "cascade_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    with open(OUT_DIR / "best_thresholds.json", "w") as f:
        json.dump({"tau_low": best["tau_low"], "tau_high": best["tau_high"],
                   "fusion_model": str(fusion_path)}, f, indent=2)
    print(f"\nSaved → {OUT_DIR/'cascade_results.json'}, {OUT_DIR/'best_thresholds.json'}, {fusion_path}")


if __name__ == "__main__":
    main()
