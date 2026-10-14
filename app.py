"""BBB permeability demo: RF-gated GIN cascade.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.models.hybrid_cascade import HybridCascade

THRESHOLDS_PATH = Path("results/hybrid/best_thresholds.json")

EXAMPLES = {
    "BBB+ (confident)": "C1=C(Cl)C=CC3=C1N(C2=C(C=CC=C2)S3)CCCN5CCN(CCN4C(N(C)CC4)=O)CC5",
    "BBB- (confident)": "CCN1C=C(C(O)=O)C(=O)c2cc(F)c(nc12)N3CCNCC3",
    "Borderline (GIN verifies)": "CC(=O)N1CCN(CC1)c2ccc(OC[C@H]3CO[C@@](Cn4ccnc4)(O3)c5ccc(Cl)cc5Cl)cc2",
}

st.set_page_config(page_title="BBB Permeability — RF + GIN Cascade", layout="wide")


@st.cache_resource
def load_cascade() -> HybridCascade:
    cfg = json.loads(THRESHOLDS_PATH.read_text())
    return HybridCascade(
        tau_low=cfg["tau_low"],
        tau_high=cfg["tau_high"],
        fusion="soft",
        fusion_model_path=cfg["fusion_model"],
    )


def mol_image(smiles: str, size: tuple[int, int] = (420, 420)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def verdict(prob: float, thr: float = 0.5) -> str:
    return "BBB+  (permeable)" if prob >= thr else "BBB-  (non-permeable)"


cascade = load_cascade()
st.title("Blood-Brain Barrier Permeability")
st.caption(
    f"Two-stage cascade — Random Forest screens every molecule, "
    f"GIN verifies only uncertain cases "
    f"(RF prob in [{cascade.tau_low:.2f}, {cascade.tau_high:.2f}]). "
    f"Test: cascade-soft F1 0.9275, acc 0.8856, GIN called on 14.7%."
)

tab_single, tab_batch = st.tabs(["Single molecule", "Batch scoring"])

# ---------------- Tab 1: single molecule ----------------
with tab_single:
    if "smiles" not in st.session_state:
        st.session_state.smiles = EXAMPLES["BBB+ (confident)"]

    cols = st.columns(3)
    for col, (label, smi) in zip(cols, EXAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state.smiles = smi

    with st.form("predict_form"):
        smiles = st.text_input("SMILES", key="smiles")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted and smiles.strip():
        try:
            t0 = time.perf_counter()
            out = cascade.predict_smiles_detailed([smiles.strip()])
            dt = time.perf_counter() - t0
        except ValueError as e:
            st.error(f"Invalid input: {e}")
            st.stop()

        rf_p = float(out["rf_prob"][0])
        gin_p = out["gin_prob"][0]
        final_p = float(out["final_prob"][0])
        routed = bool(out["routed"][0])

        left, right = st.columns([1, 1])
        with left:
            img = mol_image(smiles.strip())
            if img is not None:
                st.image(img, caption="2D structure")
            st.metric("Prediction", verdict(final_p), f"{final_p:.3f} probability")
            st.caption(f"Inference time: {dt * 1000:.0f} ms")
        with right:
            st.subheader("Stage trace")
            st.write(f"**Stage 1 — Random Forest:** {rf_p:.3f}")
            st.progress(min(max(rf_p, 0.0), 1.0))
            if routed:
                st.info(
                    f"RF uncertain ({cascade.tau_low:.2f} ≤ {rf_p:.3f} ≤ {cascade.tau_high:.2f}) "
                    f"→ routed to GIN verifier."
                )
                st.write(f"**Stage 2 — GIN verifier:** {gin_p:.3f}")
                st.progress(min(max(float(gin_p), 0.0), 1.0))
                st.write(f"**Fused result (soft):** {final_p:.3f}")
            else:
                st.success(
                    f"RF confident ({rf_p:.3f} outside "
                    f"[{cascade.tau_low:.2f}, {cascade.tau_high:.2f}]) "
                    f"→ GIN skipped, RF stands."
                )
            st.divider()
            st.write("**Model comparison**")
            cmp = pd.DataFrame(
                {
                    "model": ["RF", "GIN", "Cascade (final)"],
                    "probability": [rf_p, gin_p, final_p],
                    "call": ["always", "routed only", "—"],
                }
            )
            st.dataframe(cmp, hide_index=True, use_container_width=True)

# ---------------- Tab 2: batch ----------------
with tab_batch:
    st.write("Score many molecules at once. Provide a CSV with a `smiles` column, or paste SMILES (one per line).")
    up = st.file_uploader("CSV file", type=["csv"])
    pasted = st.text_area("…or paste SMILES", height=120, placeholder="CCO\nc1ccccc1")

    inputs: list[str] = []
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            col = "smiles" if "smiles" in df_in.columns else df_in.columns[0]
            inputs = df_in[col].dropna().astype(str).tolist()
            st.caption(f"Loaded {len(inputs)} SMILES from column `{col}`.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    elif pasted.strip():
        inputs = [ln.strip() for ln in pasted.splitlines() if ln.strip()]

    if inputs:
        inputs = inputs[:200]
        if st.button(f"Score {len(inputs)} molecules", type="primary"):
            try:
                t0 = time.perf_counter()
                out = cascade.predict_smiles_detailed(inputs)
                dt = time.perf_counter() - t0
            except ValueError as e:
                st.error(f"Invalid input: {e}")
                st.stop()

            df = pd.DataFrame(
                {
                    "smiles": inputs,
                    "rf_prob": out["rf_prob"].round(4),
                    "routed_to_gin": out["routed"],
                    "gin_prob": pd.Series(out["gin_prob"]).round(4),
                    "final_prob": out["final_prob"].round(4),
                    "prediction": ["BBB+" if p >= 0.5 else "BBB-" for p in out["final_prob"]],
                }
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Molecules", len(df))
            c2.metric("Routed to GIN", f"{out['routed'].mean():.1%}")
            c3.metric("Predicted BBB+", f"{(out['final_prob'] >= 0.5).mean():.1%}")
            c4.metric("Total time", f"{dt:.1f} s")
            st.dataframe(df, hide_index=True, use_container_width=True)
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button(
                "Download results (CSV)", buf.getvalue(), "bbb_cascade_results.csv", "text/csv"
            )

st.divider()
st.caption(
    "RF: 500-tree ECFP4 baseline (test AUC 0.890) · GIN: 2-layer, 128-dim (test AUC 0.908) · "
    "Frozen scaffold split, 306-molecule test set. Invalid SMILES are rejected with an error."
)
