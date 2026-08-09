"""
HABIT — Interactive Soil Water Retention Predictor
Streamlit App with ONNX Runtime backend

Downloads ONNX ensemble weights from HuggingFace on first run,
then predicts water retention curves from user-supplied soil properties.
"""

import os
import io
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

HF_REPO_ID = "Teamrat/habit"
NUM_MEMBERS = 20

SCALER_PARAMS = {
    "texture": {"center": [0.2712, 0.413, 0.172], "scale": [0.456, 0.4543, 0.183]},
    "bd": {"center": [1.4], "scale": [0.31]},
    "oc": {"center": [1.28], "scale": [1.9902]},
    "ksat": {"center": [2.1206], "scale": [1.5133]},
}

EXAMPLE_SOILS = {
    "Clay (heavy)": {"sand": 10, "silt": 30, "clay": 60, "bd": 1.2, "oc": 2.0, "ksat": 5},
    "Sandy loam": {"sand": 65, "silt": 25, "clay": 10, "bd": 1.5, "oc": 0.5, "ksat": 200},
    "Silt loam": {"sand": 15, "silt": 65, "clay": 20, "bd": 1.3, "oc": 1.5, "ksat": 25},
    "Loam (average)": {"sand": 40, "silt": 40, "clay": 20, "bd": 1.35, "oc": 1.2, "ksat": 50},
    "Sand (texture only)": {"sand": 90, "silt": 5, "clay": 5, "bd": None, "oc": None, "ksat": None},
}


# ═══════════════════════════════════════════════════════════════════════════
# Load ONNX ensemble (cached across reruns)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading HABIT ensemble models...")
def load_ensemble():
    """Download ONNX weights from HuggingFace and create inference sessions."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "habit-ptf", "onnx")
    os.makedirs(cache_dir, exist_ok=True)

    sessions = []
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = 1

    for i in range(1, NUM_MEMBERS + 1):
        name = f"member_{i:02d}.onnx"
        local_path = os.path.join(cache_dir, name)

        if not os.path.exists(local_path):
            # Try local onnx_weights/ directory first (for bundled deployment)
            bundled = os.path.join(os.path.dirname(__file__), "onnx_weights", name)
            if os.path.exists(bundled):
                shutil.copy2(bundled, local_path)
            else:
                downloaded = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=f"onnx/{name}",
                )
                shutil.copy2(downloaded, local_path)

        session = ort.InferenceSession(
            local_path, sess_options, providers=["CPUExecutionProvider"]
        )
        sessions.append(session)

    return sessions


# ═══════════════════════════════════════════════════════════════════════════
# Prediction logic
# ═══════════════════════════════════════════════════════════════════════════

def robust_scale(values, center, scale):
    return ((values - np.array(center)) / np.array(scale)).astype(np.float32)


def prepare_inputs(sand, silt, clay, bd, oc, ksat, wp_kpa):
    """Prepare scaled model inputs from raw soil properties."""
    # Normalise texture
    sand_f, silt_f, clay_f = float(sand), float(silt), float(clay)
    if sand_f + silt_f + clay_f > 5:  # percentages
        sand_f, silt_f, clay_f = sand_f / 100, silt_f / 100, clay_f / 100
    total = sand_f + silt_f + clay_f
    sand_f, silt_f, clay_f = sand_f / total, silt_f / total, clay_f / total

    texture_sc = robust_scale(
        np.array([[sand_f, silt_f, clay_f]]),
        SCALER_PARAMS["texture"]["center"],
        SCALER_PARAMS["texture"]["scale"],
    )

    mask = np.zeros((1, 4), dtype=np.float32)
    mask[0, 0] = 1.0  # texture always

    if bd is not None and bd > 0:
        bd_sc = robust_scale(
            np.array([[float(bd)]]),
            SCALER_PARAMS["bd"]["center"],
            SCALER_PARAMS["bd"]["scale"],
        )
        mask[0, 1] = 1.0
    else:
        bd_sc = np.zeros((1, 1), dtype=np.float32)

    if oc is not None and oc > 0:
        oc_val = float(oc)
        if oc_val > 1.0:
            oc_val /= 100
        oc_log = np.log1p(oc_val)
        oc_sc = robust_scale(
            np.array([[oc_log]]),
            SCALER_PARAMS["oc"]["center"],
            SCALER_PARAMS["oc"]["scale"],
        )
        mask[0, 2] = 1.0
    else:
        oc_sc = np.zeros((1, 1), dtype=np.float32)

    if ksat is not None and ksat > 0:
        ksat_log = np.log10(max(float(ksat), 1e-6))
        ksat_sc = robust_scale(
            np.array([[ksat_log]]),
            SCALER_PARAMS["ksat"]["center"],
            SCALER_PARAMS["ksat"]["scale"],
        )
        mask[0, 3] = 1.0
    else:
        ksat_sc = np.zeros((1, 1), dtype=np.float32)

    wp_log = np.log10(wp_kpa).astype(np.float32).reshape(1, -1)

    feed = {
        "texture": texture_sc,
        "bd": bd_sc,
        "oc": oc_sc,
        "ksat": ksat_sc,
        "mask": mask,
        "water_potential": wp_log,
    }
    return feed, mask


def get_stage_label(mask):
    """Return human-readable stage label from mask."""
    stage_names = {
        (1, 0, 0, 0): "Stage 0 — texture only",
        (1, 1, 0, 0): "Stage 1 — texture + BD",
        (1, 1, 1, 0): "Stage 2 — texture + BD + OC",
        (1, 1, 1, 1): "Stage 3 — all properties",
    }
    mask_key = tuple(int(m) for m in mask[0])
    return stage_names.get(mask_key, f"Custom mask: {mask_key}")


def run_ensemble(sessions, feed):
    """Run all ensemble members and return stacked predictions."""
    all_preds = []
    for session in sessions:
        pred = session.run(None, feed)[0]
        all_preds.append(pred[0])
    return np.array(all_preds)


def make_plot(wp_kpa, all_preds, mean, lower, upper, stage_label):
    """Create the water retention curve plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        wp_kpa, lower, upper, alpha=0.25, color="#2196F3", label="95% interval"
    )
    ax.plot(wp_kpa, mean, color="#1565C0", linewidth=2, label="Ensemble mean")
    for m in range(len(all_preds)):
        ax.plot(wp_kpa, all_preds[m], color="#90CAF9", linewidth=0.4, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Water potential |ψ| (kPa)", fontsize=12)
    ax.set_ylabel("Volumetric water content θ (cm³/cm³)", fontsize=12)
    ax.set_title(f"HABIT Prediction — {stage_label}", fontsize=13)
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="HABIT — Soil Water Retention Predictor",
    page_icon="💧",
    layout="wide",
)

st.title("💧 HABIT — Soil Water Retention Predictor")
st.markdown(
    """**Hierarchical Attention-Based Inference with Transfer Learning**

Predict soil water retention curves from basic soil properties using a 20-member
deep learning ensemble. Provide whatever properties you have — the model adapts automatically.

*Ghezzehei TA (2025). Water Resources Research.*
&nbsp;|&nbsp; [Model weights](https://huggingface.co/Teamrat/habit)
&nbsp;|&nbsp; `pip install habit-ptf`
"""
)

# Load ensemble once
ENSEMBLE = load_ensemble()

tab1, tab2, tab3 = st.tabs(["Single Soil", "Batch (CSV Upload)", "About"])

# ── Tab 1: Single soil ─────────────────────────────────────────────────
with tab1:
    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.subheader("Soil Properties")

        preset = st.selectbox(
            "Load example soil",
            ["— select —"] + list(EXAMPLE_SOILS.keys()),
        )

        if preset != "— select —" and preset in EXAMPLE_SOILS:
            ex = EXAMPLE_SOILS[preset]
            default_sand, default_silt, default_clay = ex["sand"], ex["silt"], ex["clay"]
            default_bd = ex["bd"] if ex["bd"] else 0.0
            default_oc = ex["oc"] if ex["oc"] else 0.0
            default_ksat = ex["ksat"] if ex["ksat"] else 0.0
        else:
            default_sand, default_silt, default_clay = 40.0, 40.0, 20.0
            default_bd, default_oc, default_ksat = 1.35, 1.2, 50.0

        st.markdown("**Texture** (required — % or fraction)")
        c1, c2, c3 = st.columns(3)
        sand_in = c1.number_input("Sand", value=default_sand, format="%.1f", key="sand")
        silt_in = c2.number_input("Silt", value=default_silt, format="%.1f", key="silt")
        clay_in = c3.number_input("Clay", value=default_clay, format="%.1f", key="clay")

        st.markdown("**Optional properties** (leave at 0 to omit)")
        bd_in = st.number_input("Bulk density (g/cm³)", value=default_bd, format="%.2f", key="bd")
        oc_in = st.number_input("Organic carbon (%)", value=default_oc, format="%.2f", key="oc")
        ksat_in = st.number_input("Ksat (cm/day)", value=default_ksat, format="%.1f", key="ksat")

        st.markdown("**Water potential range**")
        rc1, rc2, rc3 = st.columns(3)
        wp_min = rc1.number_input("Min (kPa)", value=1, key="wp_min")
        wp_max = rc2.number_input("Max (kPa)", value=15000, key="wp_max")
        n_pts = rc3.number_input("Points", value=50, key="n_pts")

        predict_btn = st.button("🔬 Predict", type="primary", use_container_width=True)

    with col_output:
        if predict_btn:
            # Validate
            if sand_in + silt_in + clay_in < 1:
                st.error("Texture fractions must sum to ~100% (or ~1.0).")
            else:
                wp_kpa = np.logspace(
                    np.log10(max(float(wp_min), 0.1)),
                    np.log10(float(wp_max)),
                    int(n_pts),
                )

                bd_val = bd_in if bd_in > 0 else None
                oc_val = oc_in if oc_in > 0 else None
                ksat_val = ksat_in if ksat_in > 0 else None

                feed, mask = prepare_inputs(
                    sand_in, silt_in, clay_in, bd_val, oc_val, ksat_val, wp_kpa
                )
                stage_label = get_stage_label(mask)

                with st.spinner(f"Running {NUM_MEMBERS}-member ensemble..."):
                    all_preds = run_ensemble(ENSEMBLE, feed)

                mean = np.mean(all_preds, axis=0)
                std = np.std(all_preds, axis=0)
                lower = np.percentile(all_preds, 2.5, axis=0)
                upper = np.percentile(all_preds, 97.5, axis=0)

                # Plot
                fig = make_plot(wp_kpa, all_preds, mean, lower, upper, stage_label)
                st.pyplot(fig)
                plt.close(fig)

                # Table at standard tensions
                standard_kpa = [1, 3, 6, 10, 33, 100, 300, 500, 1000, 5000, 10000, 15000]
                standard_kpa = [p for p in standard_kpa if float(wp_min) <= p <= float(wp_max)]

                table_rows = []
                for target_kpa in standard_kpa:
                    idx = np.argmin(np.abs(wp_kpa - target_kpa))
                    table_rows.append({
                        "ψ (kPa)": int(target_kpa),
                        "θ mean": f"{mean[idx]:.4f}",
                        "θ std": f"{std[idx]:.4f}",
                        "θ lower 95%": f"{lower[idx]:.4f}",
                        "θ upper 95%": f"{upper[idx]:.4f}",
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

                # CSV download
                full_df = pd.DataFrame({
                    "water_potential_kPa": wp_kpa,
                    "water_content_mean": mean,
                    "water_content_std": std,
                    "water_content_lower95": lower,
                    "water_content_upper95": upper,
                })
                for m in range(len(all_preds)):
                    full_df[f"member_{m + 1:02d}"] = all_preds[m]

                csv_buf = io.StringIO()
                full_df.to_csv(csv_buf, index=False)
                st.download_button(
                    "📥 Download full results (CSV)",
                    csv_buf.getvalue(),
                    file_name="habit_prediction.csv",
                    mime="text/csv",
                )

# ── Tab 2: Batch from CSV ──────────────────────────────────────────────
with tab2:
    st.subheader("Batch Prediction")
    st.markdown(
        """Upload a CSV with columns: `sand`, `silt`, `clay` (required),
plus optional `bd`, `oc`, `ksat`, `soil_id`.

Values can be percentages (0–100) or fractions (0–1). Missing optional
properties should be blank or 0."""
    )

    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    batch_btn = st.button("🔬 Predict All", type="primary", key="batch")

    if batch_btn and csv_file is not None:
        df = pd.read_csv(csv_file)
        cols_lower = {c.lower(): c for c in df.columns}

        if not all(k in cols_lower for k in ["sand", "silt", "clay"]):
            st.error(f"CSV must have sand, silt, clay columns. Found: {list(df.columns)}")
        else:
            wp_kpa = np.logspace(np.log10(1), np.log10(15000), 50)
            results_all = []
            progress = st.progress(0, text="Processing soils...")

            for idx, row in df.iterrows():
                sand = row[cols_lower["sand"]]
                silt = row[cols_lower["silt"]]
                clay = row[cols_lower["clay"]]
                bd = row.get(cols_lower.get("bd")) if "bd" in cols_lower else None
                oc = row.get(cols_lower.get("oc")) if "oc" in cols_lower else None
                ksat = row.get(cols_lower.get("ksat")) if "ksat" in cols_lower else None
                soil_id = row.get(cols_lower.get("soil_id", ""), idx + 1)

                bd = None if bd is not None and (pd.isna(bd) or bd <= 0) else bd
                oc = None if oc is not None and (pd.isna(oc) or oc <= 0) else oc
                ksat = None if ksat is not None and (pd.isna(ksat) or ksat <= 0) else ksat

                feed, mask = prepare_inputs(sand, silt, clay, bd, oc, ksat, wp_kpa)
                all_preds = run_ensemble(ENSEMBLE, feed)
                mean = np.mean(all_preds, axis=0)
                std = np.std(all_preds, axis=0)
                lower = np.percentile(all_preds, 2.5, axis=0)
                upper = np.percentile(all_preds, 97.5, axis=0)

                pred_df = pd.DataFrame({
                    "soil_id": soil_id,
                    "water_potential_kPa": wp_kpa,
                    "water_content_mean": mean,
                    "water_content_std": std,
                    "water_content_lower95": lower,
                    "water_content_upper95": upper,
                })
                results_all.append(pred_df)
                progress.progress((idx + 1) / len(df), text=f"Processed {idx + 1}/{len(df)} soils")

            combined = pd.concat(results_all, ignore_index=True)
            progress.empty()

            # Summary
            summary = (
                combined.groupby("soil_id")
                .agg(
                    n_points=("water_content_mean", "count"),
                    theta_sat=("water_content_mean", "max"),
                    theta_15000=("water_content_mean", "min"),
                )
                .reset_index()
            )
            st.success(f"Predicted {len(df)} soils successfully.")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            csv_buf = io.StringIO()
            combined.to_csv(csv_buf, index=False)
            st.download_button(
                "📥 Download batch results (CSV)",
                csv_buf.getvalue(),
                file_name="habit_batch_predictions.csv",
                mime="text/csv",
            )

# ── Tab 3: About ───────────────────────────────────────────────────────
with tab3:
    st.subheader("About HABIT")
    st.markdown(
        """HABIT is a deep learning model for predicting soil water retention curves
from basic soil properties. It uses a transformer-based architecture with:

- **Property-specific encoders** for each soil property
- **Cross-attention layers** that learn interactions between properties
- **Monotonic output layer** ensuring physically correct behavior
  (water content decreases with increasing tension)
- **Hierarchical training** so one model handles any combination of inputs

#### Performance (test set, 95% CI from cluster bootstrap)

| Inputs | R² | RMSE (cm³/cm³) |
|---|---|---|
| Texture only | 0.78 [0.74, 0.82] | 0.067 |
| + Bulk density | 0.85 [0.75, 0.91] | 0.056 |
| + Organic carbon | 0.86 [0.78, 0.92] | 0.052 |
| + Ksat | 0.92 [0.90, 0.94] | 0.043 |

#### Python package

```bash
pip install habit-ptf
```

```python
from habit_ptf import load_ensemble

predictor = load_ensemble()
result = predictor.predict(soil_dataframe)
```

#### Citation

Ghezzehei TA (2025). Interpretable Soil Water Retention Prediction Using
Hierarchical Attention Networks with Uncertainty Quantification.
*Water Resources Research*.

#### License

MIT (code and weights). Training data: CC BY 4.0.
"""
    )
