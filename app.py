"""
HABIT — Interactive Soil Water Retention Predictor
Streamlit App with ONNX Runtime backend
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import shutil

# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="HABIT — Soil Water Retention Predictor",
    page_icon="💧",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Global ────────────────────────────────────────────── */
    .block-container { padding-top: 1.5rem; }

    /* ── Header ────────────────────────────────────────────── */
    .app-header {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.2rem;
    }
    .app-header h1 {
        font-size: 1.7rem !important;
        margin: 0 !important;
        color: #0D47A1 !important;
    }
    .app-subtitle {
        color: #546E7A;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
        line-height: 1.4;
    }
    .app-subtitle a { color: #1565C0; text-decoration: none; }
    .app-subtitle a:hover { text-decoration: underline; }

    /* ── Stage badge ───────────────────────────────────────── */
    .stage-badge {
        display: inline-block;
        background: #E3F2FD;
        color: #1565C0;
        padding: 0.25rem 0.75rem;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* ── Texture sum indicator ─────────────────────────────── */
    .tex-ok   { color: #2E7D32; font-weight: 600; font-size: 0.85rem; }
    .tex-warn { color: #E65100; font-weight: 600; font-size: 0.85rem; }

    /* ── Ensemble note ─────────────────────────────────────── */
    .ens-note {
        background: #FAFAFA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.6rem 0 1rem 0;
        font-size: 0.85rem;
        color: #424242;
    }
    .ens-note b { color: #1565C0; }

    /* ── Input section card ────────────────────────────────── */
    .input-section-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #78909C;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    /* ── Footer / citation ─────────────────────────────────── */
    .app-footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E0E0E0;
        font-size: 0.82rem;
        color: #78909C;
        line-height: 1.5;
    }
    .app-footer a { color: #1565C0; text-decoration: none; }

    /* ── Cleaner number inputs ─────────────────────────────── */
    .stNumberInput > div > div > input { text-align: center; }

    /* ── Download button ───────────────────────────────────── */
    .stDownloadButton > button { width: 100%; }

    /* ── Tab styling ───────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)

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
    "Clay (heavy)":        {"sand": 10.0, "silt": 30.0, "clay": 60.0, "bd": 1.20, "oc": 2.0,  "ksat": 5.0},
    "Sandy loam":          {"sand": 65.0, "silt": 25.0, "clay": 10.0, "bd": 1.50, "oc": 0.5,  "ksat": 200.0},
    "Silt loam":           {"sand": 15.0, "silt": 65.0, "clay": 20.0, "bd": 1.30, "oc": 1.5,  "ksat": 25.0},
    "Loam (average)":      {"sand": 40.0, "silt": 40.0, "clay": 20.0, "bd": 1.35, "oc": 1.2,  "ksat": 50.0},
    "Sand (texture only)": {"sand": 90.0, "silt": 5.0,  "clay": 5.0,  "bd": None, "oc": None, "ksat": None},
}


# ═══════════════════════════════════════════════════════════════════════════
# Load ONNX ensemble (cached across reruns)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading HABIT ensemble models...")
def load_ensemble():
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
            bundled = os.path.join(os.path.dirname(__file__), "onnx_weights", name)
            if os.path.exists(bundled):
                shutil.copy2(bundled, local_path)
            else:
                downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=f"onnx/{name}")
                shutil.copy2(downloaded, local_path)
        session = ort.InferenceSession(local_path, sess_options, providers=["CPUExecutionProvider"])
        sessions.append(session)
    return sessions


# ═══════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ═══════════════════════════════════════════════════════════════════════════

def robust_scale(values, center, scale):
    return ((values - np.array(center)) / np.array(scale)).astype(np.float32)


def prepare_inputs(sand, silt, clay, bd, oc, ksat, wp_kpa):
    sand_f, silt_f, clay_f = float(sand), float(silt), float(clay)
    if sand_f + silt_f + clay_f > 5:
        sand_f, silt_f, clay_f = sand_f / 100, silt_f / 100, clay_f / 100
    total = sand_f + silt_f + clay_f
    sand_f, silt_f, clay_f = sand_f / total, silt_f / total, clay_f / total

    texture_sc = robust_scale(
        np.array([[sand_f, silt_f, clay_f]]),
        SCALER_PARAMS["texture"]["center"],
        SCALER_PARAMS["texture"]["scale"],
    )

    mask = np.zeros((1, 4), dtype=np.float32)
    mask[0, 0] = 1.0

    if bd is not None:
        bd_sc = robust_scale(np.array([[float(bd)]]), SCALER_PARAMS["bd"]["center"], SCALER_PARAMS["bd"]["scale"])
        mask[0, 1] = 1.0
    else:
        bd_sc = np.zeros((1, 1), dtype=np.float32)

    if oc is not None:
        oc_val = float(oc)
        if oc_val > 1.0:
            oc_val /= 100
        oc_log = np.log1p(oc_val)
        oc_sc = robust_scale(np.array([[oc_log]]), SCALER_PARAMS["oc"]["center"], SCALER_PARAMS["oc"]["scale"])
        mask[0, 2] = 1.0
    else:
        oc_sc = np.zeros((1, 1), dtype=np.float32)

    if ksat is not None:
        ksat_log = np.log10(max(float(ksat), 1e-6))
        ksat_sc = robust_scale(np.array([[ksat_log]]), SCALER_PARAMS["ksat"]["center"], SCALER_PARAMS["ksat"]["scale"])
        mask[0, 3] = 1.0
    else:
        ksat_sc = np.zeros((1, 1), dtype=np.float32)

    wp_log = np.log10(wp_kpa).astype(np.float32).reshape(1, -1)
    feed = {
        "texture": texture_sc, "bd": bd_sc, "oc": oc_sc,
        "ksat": ksat_sc, "mask": mask, "water_potential": wp_log,
    }
    return feed, mask


def get_stage_label(mask):
    stage_names = {
        (1, 0, 0, 0): "Stage 0 — texture only",
        (1, 1, 0, 0): "Stage 1 — texture + BD",
        (1, 1, 1, 0): "Stage 2 — texture + BD + OC",
        (1, 1, 1, 1): "Stage 3 — all properties",
    }
    return stage_names.get(tuple(int(m) for m in mask[0]), "Custom")


def run_ensemble(sessions, feed):
    return np.array([s.run(None, feed)[0][0] for s in sessions])


def make_plot(wp_kpa, all_preds, mean, lower, upper, stage_label):
    fig, ax = plt.subplots(figsize=(9, 5.2))

    # Individual members — very faint
    for m in range(len(all_preds)):
        ax.plot(wp_kpa, all_preds[m], color="#BBDEFB", linewidth=0.35, alpha=0.55)

    # Spread band
    ax.fill_between(wp_kpa, lower, upper, alpha=0.18, color="#1976D2",
                    label="2.5–97.5 percentile (20 members)")

    # Ensemble mean — prominent
    ax.plot(wp_kpa, mean, color="#0D47A1", linewidth=2.3, label="Ensemble mean", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Water potential |ψ| (kPa)", fontsize=11.5)
    ax.set_ylabel("Volumetric water content θ (cm³/cm³)", fontsize=11.5)
    ax.set_title(stage_label, fontsize=12, fontweight="bold", color="#37474F", pad=10)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92, edgecolor="#BDBDBD")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=wp_kpa[0] * 0.9, right=wp_kpa[-1] * 1.1)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.tick_params(labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
    <h1>💧 HABIT</h1>
</div>
<p class="app-subtitle">
    Predict soil water retention curves from basic soil properties using a
    20-member deep learning ensemble.
    Provide whatever properties you have — the model adapts automatically.<br>
    <a href="https://huggingface.co/Teamrat/habit">Model&nbsp;weights</a>
    &nbsp;·&nbsp; <code>pip install habit-ptf</code>
</p>
""", unsafe_allow_html=True)

# Load ensemble once
ENSEMBLE = load_ensemble()

tab1, tab2, tab3 = st.tabs(["Single Soil", "Batch (CSV)", "About"])

# ═══════════════════════════════════════════════════════════════════════════
# Tab 1 — Single Soil
# ═══════════════════════════════════════════════════════════════════════════

with tab1:

    # ── Input form (full width, compact) ──────────────────────────────────

    preset = st.selectbox(
        "Example soils",
        ["— custom —"] + list(EXAMPLE_SOILS.keys()),
        label_visibility="collapsed",
        key="preset_sel",
    )

    if preset != "— custom —" and preset in EXAMPLE_SOILS:
        ex = EXAMPLE_SOILS[preset]
        d_sand, d_silt, d_clay = float(ex["sand"]), float(ex["silt"]), float(ex["clay"])
        d_bd   = float(ex["bd"])   if ex["bd"]   is not None else None
        d_oc   = float(ex["oc"])   if ex["oc"]   is not None else None
        d_ksat = float(ex["ksat"]) if ex["ksat"] is not None else None
    else:
        d_sand, d_silt, d_clay = 40.0, 40.0, 20.0
        d_bd, d_oc, d_ksat = 1.35, 1.2, 50.0

    # Texture row
    st.markdown('<p class="input-section-label">Texture (required — % or fraction)</p>',
                unsafe_allow_html=True)
    tc1, tc2, tc3, tc4 = st.columns([1, 1, 1, 1.2])
    sand_in = tc1.number_input("Sand", value=d_sand, format="%.1f", key="sand")
    silt_in = tc2.number_input("Silt", value=d_silt, format="%.1f", key="silt")
    clay_in = tc3.number_input("Clay", value=d_clay, format="%.1f", key="clay")

    # Live texture sum check
    tex_sum = sand_in + silt_in + clay_in
    if tex_sum > 5:
        tex_pct = tex_sum
    else:
        tex_pct = tex_sum * 100

    if abs(tex_pct - 100) < 2:
        tc4.markdown(f'<br><span class="tex-ok">Sum: {tex_pct:.1f}%&ensp;✓</span>',
                     unsafe_allow_html=True)
    else:
        tc4.markdown(f'<br><span class="tex-warn">Sum: {tex_pct:.1f}%&ensp;≠ 100</span>',
                     unsafe_allow_html=True)

    # Optional properties row
    st.markdown('<p class="input-section-label">Optional properties (leave empty to omit)</p>',
                unsafe_allow_html=True)
    oc1, oc2, oc3 = st.columns(3)
    bd_in   = oc1.number_input("Bulk density (g/cm³)", value=d_bd,   format="%.2f",
                               key="bd",   min_value=0.0, placeholder="—")
    oc_in   = oc2.number_input("Organic carbon (%)",   value=d_oc,   format="%.2f",
                               key="oc",   min_value=0.0, placeholder="—")
    ksat_in = oc3.number_input("Ksat (cm/day)",        value=d_ksat, format="%.1f",
                               key="ksat", min_value=0.0, placeholder="—")

    # Water potential + predict button row
    st.markdown('<p class="input-section-label">Water potential range</p>',
                unsafe_allow_html=True)
    wc1, wc2, wc3, wc4 = st.columns([1, 1, 1, 1.2])
    wp_min = wc1.number_input("Min (kPa)",  value=1.0,     format="%.1f", key="wp_min")
    wp_max = wc2.number_input("Max (kPa)",  value=15000.0, format="%.0f", key="wp_max")
    n_pts  = wc3.number_input("Points",     value=50,       key="n_pts", min_value=10, max_value=200)
    wc4.markdown("<br>", unsafe_allow_html=True)
    predict_btn = wc4.button("Predict", type="primary", use_container_width=True)

    # ── Results (full width, below inputs) ────────────────────────────────

    if predict_btn:
        if abs(tex_pct - 100) >= 10:
            st.error(f"Texture fractions sum to {tex_pct:.1f}% — must be close to 100%.")
        else:
            wp_kpa = np.logspace(
                np.log10(max(float(wp_min), 0.1)),
                np.log10(float(wp_max)),
                int(n_pts),
            )

            feed, mask = prepare_inputs(
                sand_in, silt_in, clay_in, bd_in, oc_in, ksat_in, wp_kpa
            )
            stage_label = get_stage_label(mask)

            with st.spinner(f"Running {NUM_MEMBERS}-member ensemble..."):
                all_preds = run_ensemble(ENSEMBLE, feed)

            mean  = np.mean(all_preds, axis=0)
            std   = np.std(all_preds, axis=0)
            lower = np.percentile(all_preds, 2.5, axis=0)
            upper = np.percentile(all_preds, 97.5, axis=0)

            st.markdown("---")

            # Stage badge + ensemble note — side by side
            r1, r2 = st.columns([2, 1])
            r1.markdown(f'<span class="stage-badge">{stage_label}</span>',
                        unsafe_allow_html=True)
            mean_disagree = np.mean(std)
            r2.markdown(
                f'<span style="font-size:0.85rem; color:#546E7A;">'
                f'Mean ensemble σ&nbsp;=&nbsp;{mean_disagree:.4f} cm³/cm³</span>',
                unsafe_allow_html=True,
            )

            # Plot — full width
            fig = make_plot(wp_kpa, all_preds, mean, lower, upper, stage_label)
            st.pyplot(fig)
            plt.close(fig)

            # Ensemble note
            st.markdown(
                '<div class="ens-note">'
                '<b>Ensemble spread</b> reflects disagreement among 20 independently '
                'trained models. It is not a calibrated uncertainty interval, but in '
                'held-out test data, larger spread was empirically associated with '
                'larger prediction error.'
                '</div>',
                unsafe_allow_html=True,
            )

            # Table + download side by side
            col_t, col_d = st.columns([3, 1])

            standard_kpa = [1, 3, 6, 10, 33, 100, 300, 500, 1000, 5000, 10000, 15000]
            standard_kpa = [p for p in standard_kpa if float(wp_min) <= p <= float(wp_max)]

            table_rows = []
            for target_kpa in standard_kpa:
                idx = np.argmin(np.abs(wp_kpa - target_kpa))
                table_rows.append({
                    "ψ (kPa)": int(target_kpa),
                    "θ mean": f"{mean[idx]:.4f}",
                    "σ": f"{std[idx]:.4f}",
                    "2.5%": f"{lower[idx]:.4f}",
                    "97.5%": f"{upper[idx]:.4f}",
                })
            col_t.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True,
            )

            # CSV download
            full_df = pd.DataFrame({
                "water_potential_kPa": wp_kpa,
                "water_content_mean": mean,
                "water_content_std": std,
                "water_content_q025": lower,
                "water_content_q975": upper,
            })
            for m in range(len(all_preds)):
                full_df[f"member_{m + 1:02d}"] = all_preds[m]

            csv_buf = io.StringIO()
            full_df.to_csv(csv_buf, index=False)
            col_d.markdown("<br>", unsafe_allow_html=True)
            col_d.download_button(
                "📥 CSV",
                csv_buf.getvalue(),
                file_name="habit_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ═══════════════════════════════════════════════════════════════════════════
# Tab 2 — Batch CSV
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("#### Batch Prediction")
    st.markdown(
        "Upload a CSV with columns: `sand`, `silt`, `clay` (required), "
        "plus optional `bd`, `oc`, `ksat`, `soil_id`. "
        "Values can be percentages (0–100) or fractions (0–1). "
        "Missing optional properties should be blank (empty cells)."
    )

    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    batch_btn = st.button("Predict All", type="primary", key="batch")

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
                bd   = row.get(cols_lower.get("bd"))   if "bd"   in cols_lower else None
                oc   = row.get(cols_lower.get("oc"))   if "oc"   in cols_lower else None
                ksat = row.get(cols_lower.get("ksat")) if "ksat" in cols_lower else None
                soil_id = row.get(cols_lower.get("soil_id", ""), idx + 1)

                bd   = None if bd   is not None and pd.isna(bd)   else bd
                oc   = None if oc   is not None and pd.isna(oc)   else oc
                ksat = None if ksat is not None and pd.isna(ksat) else ksat

                feed, mask = prepare_inputs(sand, silt, clay, bd, oc, ksat, wp_kpa)
                all_preds = run_ensemble(ENSEMBLE, feed)
                mean  = np.mean(all_preds, axis=0)
                std   = np.std(all_preds, axis=0)
                lower = np.percentile(all_preds, 2.5, axis=0)
                upper = np.percentile(all_preds, 97.5, axis=0)

                pred_df = pd.DataFrame({
                    "soil_id": soil_id,
                    "water_potential_kPa": wp_kpa,
                    "water_content_mean": mean,
                    "water_content_std": std,
                    "water_content_q025": lower,
                    "water_content_q975": upper,
                })
                results_all.append(pred_df)
                progress.progress((idx + 1) / len(df), text=f"Processed {idx + 1}/{len(df)} soils")

            combined = pd.concat(results_all, ignore_index=True)
            progress.empty()

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
                "📥  Download batch results (CSV)",
                csv_buf.getvalue(),
                file_name="habit_batch_predictions.csv",
                mime="text/csv",
            )

# ═══════════════════════════════════════════════════════════════════════════
# Tab 3 — About
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("#### About HABIT")
    st.markdown("""
HABIT (**H**ierarchical **A**ttention-**B**ased **I**nference with **T**ransfer
learning) is a deep learning model for predicting soil water retention curves
from basic soil properties. It uses a transformer-based architecture with
property-specific encoders, cross-attention layers that learn interactions
between properties, a monotonic output layer enforcing physically correct
behavior, and hierarchical training so one model handles any combination
of inputs.
""")

    st.markdown("#### Performance")
    st.markdown("Independent test set, 95% CI from cluster bootstrap (1,000 iterations):")

    perf_df = pd.DataFrame({
        "Inputs":         ["Texture only", "+ Bulk density", "+ Organic carbon", "+ Ksat"],
        "R²":             ["0.779 [0.737, 0.817]", "0.846 [0.748, 0.906]",
                           "0.862 [0.781, 0.920]", "0.923 [0.899, 0.944]"],
        "RMSE (cm³/cm³)": ["0.067 [0.060, 0.074]", "0.056 [0.044, 0.070]",
                           "0.052 [0.040, 0.066]", "0.043 [0.036, 0.050]"],
        "MAE (cm³/cm³)":  ["0.049 [0.044, 0.055]", "0.039 [0.033, 0.049]",
                           "0.038 [0.030, 0.047]", "0.030 [0.026, 0.035]"],
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("#### Ensemble spread")
    st.markdown("""
The shaded band and σ values in the predictions represent the spread among 20
independently trained models. This is *not* a calibrated uncertainty interval.
However, in the independent HABIT test set, ensemble spread was empirically
associated with prediction error and can be used as an indicator of prediction
reliability.
""")

    st.markdown("#### Python package")
    st.code("pip install habit-ptf", language="bash")
    st.code("""from habit_ptf import load_ensemble

predictor = load_ensemble()
result = predictor.predict(soil_dataframe)""", language="python")

    st.markdown("#### License")
    st.markdown("MIT (code and weights). Training data: CC BY 4.0.")

# ═══════════════════════════════════════════════════════════════════════════
# Footer — citation lives here, not in the header
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-footer">
    Ghezzehei,&nbsp;T.A.&nbsp;(2026).
    Interpretable soil water retention prediction using hierarchical attention
    networks with uncertainty quantification.
    <i>Water Resources Research</i>, 62, e2025WR042833.
    <a href="https://doi.org/10.1029/2025WR042833">doi:10.1029/2025WR042833</a>
</div>
""", unsafe_allow_html=True)
