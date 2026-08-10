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

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HABIT — Soil Water Retention",
    page_icon="💧",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Global */
.block-container { padding-top: 1rem; max-width: 1400px; }

/* Header */
.app-title {
    font-size: 1.5rem; font-weight: 700; color: #1e293b;
    margin: 0 0 0.1rem 0;
}
.app-tagline {
    font-size: 0.85rem; color: #64748b; margin-bottom: 0.6rem; line-height: 1.5;
}
.app-tagline a { color: #3b82f6; text-decoration: none; }
.app-tagline a:hover { text-decoration: underline; }

/* Section labels */
.section-label {
    font-size: 0.7rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 0.9rem 0 0.15rem 0;
}

/* Texture sum */
.tex-ok   { color: #16a34a; font-size: 0.82rem; font-weight: 600; }
.tex-warn { color: #ea580c; font-size: 0.82rem; font-weight: 600; }

/* Stage badge */
.stage-badge {
    display: inline-block;
    background: #eff6ff; color: #3b82f6;
    padding: 0.2rem 0.65rem; border-radius: 20px;
    font-weight: 600; font-size: 0.8rem;
    margin-bottom: 0.4rem;
}

/* Metric row */
.metric-row { display: flex; gap: 0.6rem; margin-bottom: 0.6rem; }
.metric-card {
    flex: 1; background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 0.45rem 0.6rem; text-align: center;
}
.metric-card .val { font-size: 1rem; font-weight: 700; color: #1e293b; }
.metric-card .lbl { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; }

/* Ensemble note */
.ens-note {
    background: #f8fafc; border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    padding: 0.55rem 0.85rem; margin: 0.4rem 0 0.6rem 0;
    font-size: 0.78rem; color: #475569; line-height: 1.4;
}

/* Empty state */
.empty-state { text-align: center; padding: 6rem 2rem; color: #cbd5e1; }
.empty-state .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.empty-state p { font-size: 0.9rem; color: #94a3b8; }

/* Inputs */
.stNumberInput > div > div > input { text-align: center; }
.stDownloadButton > button { width: 100%; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 2px; }
.stTabs [data-baseweb="tab"] { padding: 8px 20px; }

/* Footer */
.app-footer {
    margin-top: 2rem; padding-top: 0.8rem;
    border-top: 1px solid #e2e8f0;
    font-size: 0.78rem; color: #94a3b8; line-height: 1.5;
}
.app-footer a { color: #3b82f6; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ── Configuration ─────────────────────────────────────────────────────────

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

# ── Load ONNX ensemble ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading HABIT ensemble models...")
def load_ensemble():
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "habit-ptf", "onnx")
    os.makedirs(cache_dir, exist_ok=True)
    sessions = []
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    for i in range(1, NUM_MEMBERS + 1):
        name = f"member_{i:02d}.onnx"
        local = os.path.join(cache_dir, name)
        if not os.path.exists(local):
            bundled = os.path.join(os.path.dirname(__file__), "onnx_weights", name)
            if os.path.exists(bundled):
                shutil.copy2(bundled, local)
            else:
                downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=f"onnx/{name}")
                shutil.copy2(downloaded, local)
        sessions.append(ort.InferenceSession(local, opts, providers=["CPUExecutionProvider"]))
    return sessions

# ── Prediction helpers ────────────────────────────────────────────────────

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
        bd_sc = robust_scale(np.array([[float(bd)]]),
                             SCALER_PARAMS["bd"]["center"], SCALER_PARAMS["bd"]["scale"])
        mask[0, 1] = 1.0
    else:
        bd_sc = np.zeros((1, 1), dtype=np.float32)

    if oc is not None:
        oc_val = float(oc)
        if oc_val > 1.0:
            oc_val /= 100
        oc_log = np.log1p(oc_val)
        oc_sc = robust_scale(np.array([[oc_log]]),
                             SCALER_PARAMS["oc"]["center"], SCALER_PARAMS["oc"]["scale"])
        mask[0, 2] = 1.0
    else:
        oc_sc = np.zeros((1, 1), dtype=np.float32)

    if ksat is not None:
        ksat_log = np.log10(max(float(ksat), 1e-6))
        ksat_sc = robust_scale(np.array([[ksat_log]]),
                               SCALER_PARAMS["ksat"]["center"], SCALER_PARAMS["ksat"]["scale"])
        mask[0, 3] = 1.0
    else:
        ksat_sc = np.zeros((1, 1), dtype=np.float32)

    wp_log = np.log10(wp_kpa).astype(np.float32).reshape(1, -1)
    return {
        "texture": texture_sc, "bd": bd_sc, "oc": oc_sc,
        "ksat": ksat_sc, "mask": mask, "water_potential": wp_log,
    }, mask


def get_stage_label(mask):
    names = {
        (1, 0, 0, 0): "Stage 0 — texture only",
        (1, 1, 0, 0): "Stage 1 — texture + BD",
        (1, 1, 1, 0): "Stage 2 — texture + BD + OC",
        (1, 1, 1, 1): "Stage 3 — all properties",
    }
    return names.get(tuple(int(m) for m in mask[0]), "Custom")


def run_ensemble(sessions, feed):
    return np.array([s.run(None, feed)[0][0] for s in sessions])


def make_plot(wp_kpa, all_preds, mean, lower, upper, stage_label):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for m in range(len(all_preds)):
        ax.plot(wp_kpa, all_preds[m], color="#93c5fd", linewidth=0.3, alpha=0.5)

    ax.fill_between(wp_kpa, lower, upper, alpha=0.15, color="#3b82f6",
                    label="2.5–97.5 percentile (20 members)")
    ax.plot(wp_kpa, mean, color="#1d4ed8", linewidth=2.2,
            label="Ensemble mean", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Water potential |ψ| (kPa)", fontsize=11)
    ax.set_ylabel("Volumetric water content θ (cm³/cm³)", fontsize=11)
    ax.set_title(stage_label, fontsize=11, fontweight="600",
                 color="#334155", pad=8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9,
              edgecolor="#e2e8f0")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=wp_kpa[0] * 0.9, right=wp_kpa[-1] * 1.1)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=9.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#cbd5e1")
    fig.tight_layout()
    return fig


# ── Header ────────────────────────────────────────────────────────────────

st.markdown("""
<p class="app-title">\U0001f4a7 HABIT</p>
<p class="app-tagline">
    Predict soil water retention curves from basic soil properties using a
    20-member deep learning ensemble.
    Provide whatever properties you have — the model adapts automatically.
    &nbsp;
    <a href="https://huggingface.co/Teamrat/habit">Weights</a>
    &nbsp;·&nbsp; <code style="font-size:0.82rem">pip install habit-ptf</code>
</p>
""", unsafe_allow_html=True)

ENSEMBLE = load_ensemble()

tab1, tab2, tab3 = st.tabs(["Single Soil", "Batch (CSV)", "About"])

# ══════════════════════════════════════════════════════════════════════════
# Tab 1 — Single Soil  (dashboard layout: inputs left, results right)
# ══════════════════════════════════════════════════════════════════════════

with tab1:
    left, right = st.columns([1, 2.5], gap="large")

    # ── Left panel: inputs ────────────────────────────────────────────────

    with left:
        # Preset selector
        preset = st.selectbox(
            "Example soils",
            ["— custom —"] + list(EXAMPLE_SOILS.keys()),
            label_visibility="collapsed",
            key="preset_sel",
        )

        # Reset widget state when preset changes
        if "last_preset" not in st.session_state:
            st.session_state.last_preset = None
        if preset != st.session_state.last_preset:
            st.session_state.last_preset = preset
            for k in ["sand", "silt", "clay",
                       "bd", "oc", "ksat",
                       "bd_on", "oc_on", "ksat_on"]:
                st.session_state.pop(k, None)

        # Determine defaults from preset
        if preset in EXAMPLE_SOILS:
            ex = EXAMPLE_SOILS[preset]
            d_sand, d_silt, d_clay = ex["sand"], ex["silt"], ex["clay"]
            d_bd, d_oc, d_ksat = ex["bd"], ex["oc"], ex["ksat"]
        else:
            d_sand, d_silt, d_clay = 40.0, 40.0, 20.0
            d_bd, d_oc, d_ksat = None, None, None

        # Texture
        st.markdown('<p class="section-label">Texture (%)</p>',
                    unsafe_allow_html=True)
        tc1, tc2, tc3 = st.columns(3)
        sand_in = tc1.number_input("Sand", value=d_sand, format="%.1f",
                                   key="sand", min_value=0.0)
        silt_in = tc2.number_input("Silt", value=d_silt, format="%.1f",
                                   key="silt", min_value=0.0)
        clay_in = tc3.number_input("Clay", value=d_clay, format="%.1f",
                                   key="clay", min_value=0.0)

        tex_sum = sand_in + silt_in + clay_in
        tex_pct = tex_sum if tex_sum > 5 else tex_sum * 100
        if abs(tex_pct - 100) < 2:
            st.markdown(f'<span class="tex-ok">✓ {tex_pct:.1f}%</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="tex-warn">⚠ {tex_pct:.1f}% ≠ 100</span>',
                        unsafe_allow_html=True)

        # Optional properties — toggles + number inputs
        st.markdown('<p class="section-label">Optional properties</p>',
                    unsafe_allow_html=True)

        bd_on = st.toggle("Bulk density (g/cm³)",
                          value=(d_bd is not None), key="bd_on")
        if bd_on:
            bd_in = st.number_input(
                "BD", value=d_bd if d_bd is not None else 1.35,
                format="%.2f", key="bd", min_value=0.01,
                label_visibility="collapsed")
        else:
            bd_in = None

        oc_on = st.toggle("Organic carbon (%)",
                          value=(d_oc is not None), key="oc_on")
        if oc_on:
            oc_in = st.number_input(
                "OC", value=d_oc if d_oc is not None else 1.2,
                format="%.2f", key="oc", min_value=0.0,
                label_visibility="collapsed")
        else:
            oc_in = None

        ksat_on = st.toggle("Kₛₐₜ (cm/day)",
                            value=(d_ksat is not None), key="ksat_on")
        if ksat_on:
            ksat_in = st.number_input(
                "Ksat", value=d_ksat if d_ksat is not None else 50.0,
                format="%.1f", key="ksat", min_value=0.0,
                label_visibility="collapsed")
        else:
            ksat_in = None

        # Water potential range
        st.markdown('<p class="section-label">Water potential range</p>',
                    unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)
        wp_min = wc1.number_input("Min (kPa)", value=1.0,
                                  format="%.1f", key="wp_min")
        wp_max = wc2.number_input("Max (kPa)", value=15000.0,
                                  format="%.0f", key="wp_max")
        n_pts = st.number_input("Points", value=50, key="n_pts",
                                min_value=10, max_value=200)

        # Predict button
        st.markdown("")
        predict_btn = st.button("Predict", type="primary",
                                use_container_width=True)

    # ── Run prediction (store in session state) ───────────────────────────

    if predict_btn:
        if abs(tex_pct - 100) >= 10:
            with right:
                st.error(f"Texture fractions sum to {tex_pct:.1f}% "
                         "— must be close to 100%.")
        else:
            wp_kpa = np.logspace(
                np.log10(max(float(wp_min), 0.1)),
                np.log10(float(wp_max)),
                int(n_pts),
            )
            feed, mask = prepare_inputs(
                sand_in, silt_in, clay_in, bd_in, oc_in, ksat_in, wp_kpa)
            stage_label = get_stage_label(mask)

            with st.spinner(f"Running {NUM_MEMBERS}-member ensemble…"):
                all_preds = run_ensemble(ENSEMBLE, feed)

            st.session_state.results = {
                "wp_kpa": wp_kpa,
                "all_preds": all_preds,
                "mean": np.mean(all_preds, axis=0),
                "std": np.std(all_preds, axis=0),
                "lower": np.percentile(all_preds, 2.5, axis=0),
                "upper": np.percentile(all_preds, 97.5, axis=0),
                "stage_label": stage_label,
                "wp_min": float(wp_min),
                "wp_max": float(wp_max),
            }

    # ── Right panel: results ──────────────────────────────────────────────

    with right:
        res = st.session_state.get("results")
        if res is not None:
            wp_kpa = res["wp_kpa"]
            all_preds = res["all_preds"]
            mean = res["mean"]
            std = res["std"]
            lower, upper = res["lower"], res["upper"]
            stage_label = res["stage_label"]

            # Stage badge
            st.markdown(
                f'<span class="stage-badge">{stage_label}</span>',
                unsafe_allow_html=True)

            # Summary metrics
            theta_sat = mean[0]
            theta_dry = mean[-1]
            mean_sigma = np.mean(std)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="val">{theta_sat:.3f}</div>
                    <div class="lbl">θ near sat.</div>
                </div>
                <div class="metric-card">
                    <div class="val">{theta_dry:.4f}</div>
                    <div class="lbl">θ at {wp_kpa[-1]:.0f} kPa</div>
                </div>
                <div class="metric-card">
                    <div class="val">{mean_sigma:.4f}</div>
                    <div class="lbl">Mean ens. σ</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Plot
            fig = make_plot(wp_kpa, all_preds, mean, lower, upper,
                            stage_label)
            st.pyplot(fig)
            plt.close(fig)

            # Ensemble note
            st.markdown(
                '<div class="ens-note">'
                '<b>Ensemble spread</b> reflects disagreement among 20 '
                'independently trained models. It is not a calibrated '
                'uncertainty interval, but larger spread was empirically '
                'associated with larger prediction error in held-out '
                'test data.'
                '</div>',
                unsafe_allow_html=True)

            # Table at standard water potentials
            r_min = res["wp_min"]
            r_max = res["wp_max"]
            standard_kpa = [p for p in
                            [1, 3, 6, 10, 33, 100, 300, 500,
                             1000, 5000, 10000, 15000]
                            if r_min <= p <= r_max]
            table_rows = []
            for kpa in standard_kpa:
                idx = np.argmin(np.abs(wp_kpa - kpa))
                table_rows.append({
                    "|ψ| (kPa)": int(kpa),
                    "θ mean": f"{mean[idx]:.4f}",
                    "σ": f"{std[idx]:.4f}",
                    "2.5%": f"{lower[idx]:.4f}",
                    "97.5%": f"{upper[idx]:.4f}",
                })

            tc, dc = st.columns([3, 1])
            tc.dataframe(pd.DataFrame(table_rows),
                         use_container_width=True, hide_index=True)

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
            dc.markdown("<br>", unsafe_allow_html=True)
            dc.download_button(
                "Download CSV",
                csv_buf.getvalue(),
                file_name="habit_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            # Empty state before first prediction
            st.markdown("""
            <div class="empty-state">
                <div class="icon">\U0001f4ca</div>
                <p>Enter soil properties and click <b>Predict</b></p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Tab 2 — Batch CSV
# ══════════════════════════════════════════════════════════════════════════

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
            st.error(f"CSV must have sand, silt, clay columns. "
                     f"Found: {list(df.columns)}")
        else:
            wp_kpa = np.logspace(np.log10(1), np.log10(15000), 50)
            results_all = []
            progress = st.progress(0, text="Processing soils…")

            for idx, row in df.iterrows():
                sand = row[cols_lower["sand"]]
                silt = row[cols_lower["silt"]]
                clay = row[cols_lower["clay"]]
                bd   = (row.get(cols_lower.get("bd"))
                        if "bd" in cols_lower else None)
                oc   = (row.get(cols_lower.get("oc"))
                        if "oc" in cols_lower else None)
                ksat = (row.get(cols_lower.get("ksat"))
                        if "ksat" in cols_lower else None)
                soil_id = row.get(cols_lower.get("soil_id", ""), idx + 1)

                bd   = None if bd   is not None and pd.isna(bd)   else bd
                oc   = None if oc   is not None and pd.isna(oc)   else oc
                ksat = None if ksat is not None and pd.isna(ksat) else ksat

                feed, mask = prepare_inputs(
                    sand, silt, clay, bd, oc, ksat, wp_kpa)
                preds = run_ensemble(ENSEMBLE, feed)
                m   = np.mean(preds, axis=0)
                s   = np.std(preds, axis=0)
                lo  = np.percentile(preds, 2.5, axis=0)
                hi  = np.percentile(preds, 97.5, axis=0)

                results_all.append(pd.DataFrame({
                    "soil_id": soil_id,
                    "water_potential_kPa": wp_kpa,
                    "water_content_mean": m,
                    "water_content_std": s,
                    "water_content_q025": lo,
                    "water_content_q975": hi,
                }))
                progress.progress((idx + 1) / len(df),
                                  text=f"Processed {idx + 1}/{len(df)} soils")

            combined = pd.concat(results_all, ignore_index=True)
            progress.empty()

            summary = (
                combined.groupby("soil_id")
                .agg(n_points=("water_content_mean", "count"),
                     theta_sat=("water_content_mean", "max"),
                     theta_15000=("water_content_mean", "min"))
                .reset_index()
            )
            st.success(f"Predicted {len(df)} soils successfully.")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            csv_buf = io.StringIO()
            combined.to_csv(csv_buf, index=False)
            st.download_button(
                "Download batch results (CSV)",
                csv_buf.getvalue(),
                file_name="habit_batch_predictions.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════
# Tab 3 — About
# ══════════════════════════════════════════════════════════════════════════

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
    st.markdown("Independent test set, 95% CI from cluster bootstrap "
                "(1,000 iterations):")

    perf_df = pd.DataFrame({
        "Inputs":         ["Texture only", "+ Bulk density",
                           "+ Organic carbon", "+ Ksat"],
        "R²":        ["0.779 [0.737, 0.817]", "0.846 [0.748, 0.906]",
                           "0.862 [0.781, 0.920]", "0.923 [0.899, 0.944]"],
        "RMSE (cm³/cm³)": [
            "0.067 [0.060, 0.074]", "0.056 [0.044, 0.070]",
            "0.052 [0.040, 0.066]", "0.043 [0.036, 0.050]"],
        "MAE (cm³/cm³)": [
            "0.049 [0.044, 0.055]", "0.039 [0.033, 0.049]",
            "0.038 [0.030, 0.047]", "0.030 [0.026, 0.035]"],
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("#### Ensemble spread")
    st.markdown("""
The shaded band and σ values in the predictions represent the spread among
20 independently trained models. This is *not* a calibrated uncertainty
interval. However, in the independent HABIT test set, ensemble spread was
empirically associated with prediction error and can be used as an indicator
of prediction reliability.
""")

    st.markdown("#### Python package")
    st.code("pip install habit-ptf", language="bash")
    st.code("""from habit_ptf import load_ensemble

predictor = load_ensemble()
result = predictor.predict(soil_dataframe)""", language="python")

    st.markdown("#### License")
    st.markdown("MIT (code and weights). Training data: CC BY 4.0.")


# ── Footer ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-footer">
    Ghezzehei,&nbsp;T.A.&nbsp;(2026).
    Interpretable soil water retention prediction using hierarchical attention
    networks with uncertainty quantification.
    <i>Water Resources Research</i>, 62, e2025WR042833.
    <a href="https://doi.org/10.1029/2025WR042833">doi:10.1029/2025WR042833</a>
</div>
""", unsafe_allow_html=True)
