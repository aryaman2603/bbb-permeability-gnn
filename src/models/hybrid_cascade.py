"""RF-gated GIN cascade: RF as fast path, GIN as verifier for uncertain cases.

Gate rule (accuracy-first, asymmetric):
    routed = (tau_low <= p_rf <= tau_high)
    final  = gin_prob (hard) or logistic([rf_prob, gin_prob]) (soft) if routed else rf_prob

GIN weights/config are loaded via src.explain.aggregate_explanations.load_model
so Optuna-tuned configs (e.g. gin hidden=128, layers=2) resolve automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.explain.aggregate_explanations import load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CascadeOutput:
    rf_prob: np.ndarray
    gin_prob: np.ndarray  # NaN where not routed
    final_prob: np.ndarray
    routed: np.ndarray  # bool mask
    y_true: np.ndarray


class HybridCascade:
    def __init__(
        self,
        rf_path: str = "results/nethra/random_forest_model.joblib",
        gin_ckpt: str = "results/nethra/gin_best.pt",
        tau_low: float = 0.30,
        tau_high: float = 0.70,
        fusion: str = "hard",
        fusion_model_path: str | None = None,
        batch_size: int = 64,
    ):
        assert 0.0 < tau_low < tau_high < 1.0, "need 0 < tau_low < tau_high < 1"
        assert fusion in ("hard", "soft")
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.fusion = fusion
        self.batch_size = batch_size

        self.rf = joblib.load(rf_path)
        self.gin = load_model("gin", gin_ckpt)
        self.gin.to(DEVICE).eval()

        self.fusion_model = None
        if fusion == "soft":
            if fusion_model_path is None:
                raise ValueError("soft fusion needs fusion_model_path (fit on val first)")
            self.fusion_model = joblib.load(fusion_model_path)

    def _gin_probs(self, data_list: list) -> np.ndarray:
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                logits = self.gin(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs) if probs else np.array([])

    def predict_data_list(
        self,
        data_list: list,
        rf_probs: np.ndarray | None = None,
    ) -> CascadeOutput:
        """Full cascade over pre-built PyG Data list.

        Pass precomputed rf_probs to avoid recomputing fingerprints (used by eval script).
        gin_prob is NaN for non-routed molecules (GIN never runs on them).
        """
        from src.features.morgan_fingerprints import build_fingerprint_dataset

        y_true = np.array([int(d.y.item()) for d in data_list], dtype=int)
        if rf_probs is None:
            X, _ = build_fingerprint_dataset(data_list)
            rf_probs = self.rf.predict_proba(X)[:, 1]

        routed = (rf_probs >= self.tau_low) & (rf_probs <= self.tau_high)
        gin_prob = np.full_like(rf_probs, np.nan, dtype=float)

        if routed.any():
            routed_data = [d for d, r in zip(data_list, routed) if r]
            gin_prob[routed] = self._gin_probs(routed_data)

        if self.fusion == "hard" or self.fusion_model is None:
            final_prob = np.where(routed, gin_prob, rf_probs)
        elif not routed.any():
            final_prob = rf_probs.copy()
        else:
            feats = np.column_stack([rf_probs[routed], gin_prob[routed]])
            fused = self.fusion_model.predict_proba(feats)[:, 1]
            final_prob = rf_probs.copy()
            final_prob[routed] = fused

        return CascadeOutput(
            rf_prob=rf_probs, gin_prob=gin_prob,
            final_prob=final_prob, routed=routed, y_true=y_true,
        )

    def predict_smiles(self, smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Single-entry inference from raw SMILES. Returns (final_prob, routed)."""
        detail = self.predict_smiles_detailed(smiles_list)
        return detail["final_prob"], detail["routed"]

    def predict_smiles_detailed(self, smiles_list: list[str]) -> dict:
        """Inference with full stage trace for UI/demo purposes.

        Returns dict with rf_prob, gin_prob (NaN when not routed),
        final_prob, routed. Raises ValueError listing invalid SMILES.
        """
        from rdkit import Chem
        from src.data.graph_builder import smiles_to_data
        from src.features.morgan_fingerprints import smiles_to_morgan

        bad = [s for s in smiles_list if Chem.MolFromSmiles(s) is None]
        if bad:
            raise ValueError(f"{len(bad)} invalid SMILES, e.g. {bad[0][:60]}")

        # RF always runs (cheap fingerprint + forest).
        X = np.stack([smiles_to_morgan(s) for s in smiles_list], axis=0)
        rf_probs = self.rf.predict_proba(X)[:, 1]
        routed = (rf_probs >= self.tau_low) & (rf_probs <= self.tau_high)
        gin_probs = np.full_like(rf_probs, np.nan, dtype=float)
        final_prob = rf_probs.copy().astype(float)

        # GIN only on routed subset.
        for i, (s, r) in enumerate(zip(smiles_list, routed)):
            if not r:
                continue
            data = smiles_to_data(s, label=0.0)
            if data is None:
                continue  # unparseable -> keep RF prob
            data = data.to(DEVICE)
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                logit = self.gin(data.x, data.edge_index, data.edge_attr, batch)
            g = torch.sigmoid(logit).cpu().item()
            gin_probs[i] = g
            if self.fusion == "hard" or self.fusion_model is None:
                final_prob[i] = g
            else:
                feats = np.array([[rf_probs[i], g]])
                final_prob[i] = float(self.fusion_model.predict_proba(feats)[0, 1])
        return {"rf_prob": rf_probs, "gin_prob": gin_probs,
                "final_prob": final_prob, "routed": routed}
