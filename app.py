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
.block-container { padding-top: 0.5rem; max-width: 1400px; }

/* Hide Streamlit hamburger menu and default header */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* Header bar */
.app-header {
    display: flex; align-items: baseline; gap: 0.7rem;
    margin: 0.3rem 0 0.6rem 0; padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}
.app-title {
    font-size: 1.6rem; font-weight: 800; color: #1e293b;
    margin: 0; letter-spacing: -0.02em;
}
.app-subtitle {
    font-size: 0.85rem; color: #64748b; margin: 0;
}

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
    background: #eff6ff; color: #1d4ed8;
    padding: 0.2rem 0.65rem; border-radius: 20px;
    font-weight: 600; font-size: 0.8rem;
    margin-bottom: 0.4rem;
}

/* Metric row */
.metric-row { display: flex; gap: 0.6rem; margin-bottom: 0.6rem; }
.metric-card {
    flex: 1; background: #f1f5f9; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 0.45rem 0.6rem; text-align: center;
}
.metric-card .val { font-size: 1rem; font-weight: 700; color: #1e293b; }
.metric-card .lbl { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; }

/* Ensemble note */
.ens-note {
    background: #f1f5f9; border-left: 3px solid #1d4ed8;
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

/* Footer */
.app-footer {
    margin-top: 2rem; padding-top: 0.8rem;
    border-top: 1px solid #e2e8f0;
    font-size: 0.78rem; color: #94a3b8; line-height: 1.5;
}
.app-footer a { color: #1d4ed8; text-decoration: none; }
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


def run_ensemble_with_attention(sessions, feed):
    """Run ensemble and collect predictions + all attention weights."""
    preds = []
    prop_attn_all = []
    cross_bd_all = []
    cross_oc_all = []
    wp_attn_all = []

    output_names = [
        "water_content",
        "property_attention_weights",
        "cross_attention_texture_bd",
        "cross_attention_texture_oc",
        "wp_attention_weights",
    ]

    for s in sessions:
        # Check if this model has attention outputs
        model_outputs = [o.name for o in s.get_outputs()]
        if "property_attention_weights" in model_outputs:
            outs = s.run(output_names, feed)
            preds.append(outs[0][0])
            prop_attn_all.append(outs[1][0])     # (num_heads, 4, 4)
            cross_bd_all.append(outs[2][0])       # (num_heads, 2, 2)
            cross_oc_all.append(outs[3][0])       # (num_heads, 2, 2)
            wp_attn_all.append(outs[4][0])        # (num_heads, 1, num_wp)
        else:
            # Fallback for old models without attention outputs
            preds.append(s.run(None, feed)[0][0])

    result = {"preds": np.array(preds)}

    if prop_attn_all:
        result["property_attention"] = np.array(prop_attn_all)
        result["cross_attention_bd"] = np.array(cross_bd_all)
        result["cross_attention_oc"] = np.array(cross_oc_all)
        result["wp_attention"] = np.array(wp_attn_all)

    return result


def prepare_batch_inputs(df, cols_lower, wp_kpa):
    """Vectorized input preparation for a batch of soils."""
    n = len(df)

    # Texture — normalize to fractions summing to 1
    sand = pd.to_numeric(df[cols_lower["sand"]], errors="coerce").values
    silt = pd.to_numeric(df[cols_lower["silt"]], errors="coerce").values
    clay = pd.to_numeric(df[cols_lower["clay"]], errors="coerce").values

    totals = sand + silt + clay
    if np.nanmean(totals) > 5:
        sand, silt, clay = sand / 100, silt / 100, clay / 100
        totals = sand + silt + clay
    sand, silt, clay = sand / totals, silt / totals, clay / totals

    texture_sc = robust_scale(
        np.column_stack([sand, silt, clay]).astype(np.float32),
        SCALER_PARAMS["texture"]["center"],
        SCALER_PARAMS["texture"]["scale"],
    )

    mask = np.zeros((n, 4), dtype=np.float32)
    mask[:, 0] = 1.0

    # BD
    if "bd" in cols_lower:
        bd_raw = pd.to_numeric(df[cols_lower["bd"]], errors="coerce").values
        bd_ok = ~np.isnan(bd_raw)
        bd_vals = np.where(bd_ok, bd_raw, 0.0).reshape(-1, 1).astype(np.float32)
        bd_sc = robust_scale(bd_vals, SCALER_PARAMS["bd"]["center"],
                             SCALER_PARAMS["bd"]["scale"])
        bd_sc[~bd_ok] = 0.0
        mask[bd_ok, 1] = 1.0
    else:
        bd_sc = np.zeros((n, 1), dtype=np.float32)

    # OC
    if "oc" in cols_lower:
        oc_raw = pd.to_numeric(df[cols_lower["oc"]], errors="coerce").values
        oc_ok = ~np.isnan(oc_raw)
        oc_vals = np.where(oc_ok, oc_raw, 0.0)
        oc_vals = np.where(oc_vals > 1.0, oc_vals / 100, oc_vals)
        oc_log = np.log1p(oc_vals).reshape(-1, 1).astype(np.float32)
        oc_sc = robust_scale(oc_log, SCALER_PARAMS["oc"]["center"],
                             SCALER_PARAMS["oc"]["scale"])
        oc_sc[~oc_ok] = 0.0
        mask[oc_ok, 2] = 1.0
    else:
        oc_sc = np.zeros((n, 1), dtype=np.float32)

    # Ksat
    if "ksat" in cols_lower:
        ksat_raw = pd.to_numeric(df[cols_lower["ksat"]], errors="coerce").values
        ksat_ok = ~np.isnan(ksat_raw)
        ksat_vals = np.where(ksat_ok, ksat_raw, 1.0)
        ksat_log = np.log10(np.maximum(ksat_vals, 1e-6)).reshape(-1, 1) \
                   .astype(np.float32)
        ksat_sc = robust_scale(ksat_log, SCALER_PARAMS["ksat"]["center"],
                               SCALER_PARAMS["ksat"]["scale"])
        ksat_sc[~ksat_ok] = 0.0
        mask[ksat_ok, 3] = 1.0
    else:
        ksat_sc = np.zeros((n, 1), dtype=np.float32)

    # Water potential — broadcast to all soils
    wp_log = np.log10(wp_kpa).astype(np.float32)
    wp_batch = np.tile(wp_log, (n, 1))

    feed = {
        "texture": texture_sc, "bd": bd_sc, "oc": oc_sc,
        "ksat": ksat_sc, "mask": mask, "water_potential": wp_batch,
    }
    return feed, mask


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
<div class="app-header">
    <p class="app-title">\U0001f4a7 HABIT</p>
    <p class="app-subtitle">Soil water retention prediction</p>
</div>
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
                ens_result = run_ensemble_with_attention(ENSEMBLE, feed)
                all_preds = ens_result["preds"]

            res_dict = {
                "wp_kpa": wp_kpa,
                "all_preds": all_preds,
                "mean": np.mean(all_preds, axis=0),
                "std": np.std(all_preds, axis=0),
                "lower": np.percentile(all_preds, 2.5, axis=0),
                "upper": np.percentile(all_preds, 97.5, axis=0),
                "stage_label": stage_label,
                "wp_min": float(wp_min),
                "wp_max": float(wp_max),
                "mask": mask,
            }

            # Attach attention weights if available
            if "property_attention" in ens_result:
                res_dict["property_attention"] = ens_result["property_attention"]
                res_dict["cross_attention_bd"] = ens_result["cross_attention_bd"]
                res_dict["cross_attention_oc"] = ens_result["cross_attention_oc"]
                res_dict["wp_attention"] = ens_result["wp_attention"]

            st.session_state.results = res_dict

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

            # ── Property attention bar chart ────────────────────────────
            if "property_attention" in res:
                # property_attention: (20, num_heads, 4, 4)
                # Average over heads, take column means (contribution of
                # each property to the combined representation)
                prop_attn = res["property_attention"]  # (20, H, 4, 4)
                head_avg = np.mean(prop_attn, axis=1)  # (20, 4, 4)
                # Column mean = how much each property contributes
                col_means = np.mean(head_avg, axis=1)  # (20, 4)
                attn_mean = np.mean(col_means, axis=0)  # (4,)
                attn_std = np.std(col_means, axis=0)     # (4,)

                prop_names = ["Texture", "Bulk density", "Organic C", "Ksat"]
                active_mask = res["mask"][0]

                fig_attn, ax_attn = plt.subplots(figsize=(8, 1.8))
                fig_attn.patch.set_facecolor("white")
                ax_attn.set_facecolor("white")

                colors = ["#3b82f6" if active_mask[i] else "#e2e8f0"
                          for i in range(4)]
                bars = ax_attn.barh(prop_names, attn_mean, xerr=attn_std,
                                    color=colors, edgecolor="white",
                                    height=0.6, capsize=3,
                                    error_kw={"linewidth": 0.8,
                                              "color": "#94a3b8"})
                ax_attn.set_xlabel("Attention weight", fontsize=9)
                ax_attn.set_xlim(0)
                ax_attn.tick_params(labelsize=9)
                for sp in ax_attn.spines.values():
                    sp.set_visible(False)
                ax_attn.grid(axis="x", alpha=0.15, linewidth=0.5)
                ax_attn.invert_yaxis()
                fig_attn.tight_layout()

                st.markdown('<p class="section-label">Property attention '
                            '(ensemble mean ± σ)</p>',
                            unsafe_allow_html=True)
                st.pyplot(fig_attn)
                plt.close(fig_attn)

                # ── Advanced attention (hidden by default) ───────────
                with st.expander("Cross-attention & water potential attention"):
                    ca_col1, ca_col2 = st.columns(2)

                    # Cross-attention texture ↔ BD
                    cross_bd = res["cross_attention_bd"]  # (20, H, 2, 2)
                    cross_bd_avg = np.mean(
                        np.mean(cross_bd, axis=1), axis=0)  # (2, 2)

                    fig_cbd, ax_cbd = plt.subplots(figsize=(3, 2.5))
                    im1 = ax_cbd.imshow(cross_bd_avg, cmap="Blues",
                                         vmin=0, vmax=1)
                    ax_cbd.set_xticks([0, 1])
                    ax_cbd.set_yticks([0, 1])
                    ax_cbd.set_xticklabels(["Texture", "BD"], fontsize=8)
                    ax_cbd.set_yticklabels(["Texture", "BD"], fontsize=8)
                    ax_cbd.set_title("Texture ↔ BD", fontsize=9)
                    for i in range(2):
                        for j in range(2):
                            ax_cbd.text(j, i, f"{cross_bd_avg[i, j]:.2f}",
                                        ha="center", va="center",
                                        fontsize=9, fontweight="bold",
                                        color="white" if cross_bd_avg[i, j] > 0.5
                                        else "#1e293b")
                    fig_cbd.tight_layout()
                    ca_col1.pyplot(fig_cbd)
                    plt.close(fig_cbd)

                    # Cross-attention texture ↔ OC
                    cross_oc = res["cross_attention_oc"]  # (20, H, 2, 2)
                    cross_oc_avg = np.mean(
                        np.mean(cross_oc, axis=1), axis=0)  # (2, 2)

                    fig_coc, ax_coc = plt.subplots(figsize=(3, 2.5))
                    im2 = ax_coc.imshow(cross_oc_avg, cmap="Oranges",
                                         vmin=0, vmax=1)
                    ax_coc.set_xticks([0, 1])
                    ax_coc.set_yticks([0, 1])
                    ax_coc.set_xticklabels(["Texture", "OC"], fontsize=8)
                    ax_coc.set_yticklabels(["Texture", "OC"], fontsize=8)
                    ax_coc.set_title("Texture ↔ OC", fontsize=9)
                    for i in range(2):
                        for j in range(2):
                            ax_coc.text(j, i, f"{cross_oc_avg[i, j]:.2f}",
                                        ha="center", va="center",
                                        fontsize=9, fontweight="bold",
                                        color="white" if cross_oc_avg[i, j] > 0.5
                                        else "#1e293b")
                    fig_coc.tight_layout()
                    ca_col2.pyplot(fig_coc)
                    plt.close(fig_coc)

                    # WP attention
                    wp_attn = res["wp_attention"]  # (20, H, 1, num_wp)
                    wp_avg = np.mean(
                        np.mean(wp_attn, axis=1), axis=0)  # (1, num_wp)
                    wp_weights = wp_avg[0]  # (num_wp,)

                    fig_wp, ax_wp = plt.subplots(figsize=(8, 2))
                    fig_wp.patch.set_facecolor("white")
                    ax_wp.set_facecolor("white")
                    ax_wp.fill_between(wp_kpa, 0, wp_weights,
                                        alpha=0.3, color="#3b82f6")
                    ax_wp.plot(wp_kpa, wp_weights, color="#1d4ed8",
                               linewidth=1.2)
                    ax_wp.set_xscale("log")
                    ax_wp.set_xlabel("Water potential |ψ| (kPa)",
                                     fontsize=9)
                    ax_wp.set_ylabel("Attention weight", fontsize=9)
                    ax_wp.set_title(
                        "Water potential attention (ensemble mean)",
                        fontsize=9)
                    ax_wp.tick_params(labelsize=8)
                    for sp in ["top", "right"]:
                        ax_wp.spines[sp].set_visible(False)
                    for sp in ["bottom", "left"]:
                        ax_wp.spines[sp].set_color("#cbd5e1")
                    fig_wp.tight_layout()
                    st.pyplot(fig_wp)
                    plt.close(fig_wp)

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

    # ── Instructions + template ───────────────────────────────────────────
    ci, ct = st.columns([2.5, 1])
    ci.markdown(
        "Upload a CSV with soil properties. The model detects which "
        "properties each row provides and adapts automatically."
    )
    template_csv = (
        "soil_id,sand,silt,clay,bd,oc,ksat\n"
        "sample_1,40,40,20,1.35,1.2,50\n"
        "sample_2,65,25,10,1.50,,\n"
        "sample_3,90,5,5,,,\n"
    )
    ct.download_button("Download template CSV", template_csv,
                       file_name="habit_template.csv", mime="text/csv",
                       use_container_width=True)

    with st.expander("CSV format details"):
        st.markdown("""
**Required columns:** `sand`, `silt`, `clay` — percentages (0–100) or
fractions (0–1), must sum to ~100% or ~1.0

**Optional columns** (leave cells empty to omit for that row):

| Column | Units | Notes |
|--------|-------|-------|
| `bd` | g/cm³ | Bulk density |
| `oc` | % by mass | Organic carbon |
| `ksat` | cm/day | Saturated hydraulic conductivity |
| `soil_id` | — | Row identifier (auto-numbered if absent) |

The model assigns each row a **stage** based on available properties:
Stage 0 (texture only) → Stage 1 (+BD) → Stage 2 (+OC) → Stage 3 (+Ksat).
Rows in the same CSV can have different stages.
""")

    # ── Water potential input ─────────────────────────────────────────────
    wp_text = st.text_input(
        "Water potentials (kPa, comma-separated)",
        value="1, 3, 6, 10, 33, 100, 300, 500, 1000, 5000, 10000, 15000",
        help="Enter one or more values. Example: '33' for field capacity only.",
    )

    # ── File upload + options ────────────────────────────────────────────
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    include_attn = st.checkbox(
        "Include property attention weights in download",
        value=False,
        help="Adds 4 columns (attn_texture, attn_bd, attn_oc, attn_ksat) "
             "showing how much the model relied on each property. "
             "Increases processing time.")
    batch_btn = st.button("Predict All", type="primary", key="batch")

    if batch_btn and csv_file is not None:
        # Parse water potentials
        try:
            wp_kpa = np.array(sorted(set(
                float(x.strip()) for x in wp_text.split(",") if x.strip()
            )))
        except ValueError:
            st.error("Could not parse water potential values. "
                     "Use comma-separated numbers (e.g. '33' or '10, 33, 1500').")
            st.stop()
        if len(wp_kpa) == 0:
            st.error("Enter at least one water potential value.")
            st.stop()

        # Read CSV
        df = pd.read_csv(csv_file)
        cols_lower = {c.strip().lower(): c for c in df.columns}

        if not all(k in cols_lower for k in ["sand", "silt", "clay"]):
            st.error(f"CSV must have sand, silt, clay columns. "
                     f"Found: {list(df.columns)}")
            st.stop()

        n_soils = len(df)
        n_wp = len(wp_kpa)

        # Prepare vectorized inputs
        feed, mask = prepare_batch_inputs(df, cols_lower, wp_kpa)

        # Chunked ensemble inference
        CHUNK = 500
        n_chunks = (n_soils + CHUNK - 1) // CHUNK
        all_mean, all_std, all_q025, all_q975 = [], [], [], []
        all_attn = [] if include_attn else None

        # Check if models support attention outputs
        attn_available = include_attn and (
            "property_attention_weights" in
            [o.name for o in ENSEMBLE[0].get_outputs()])

        progress = st.progress(0, text=f"Processing {n_soils:,} soils "
                               f"at {n_wp} water potential(s)…")

        for ci_idx in range(n_chunks):
            s = ci_idx * CHUNK
            e = min(s + CHUNK, n_soils)
            chunk_feed = {k: v[s:e] for k, v in feed.items()}

            preds = np.array(
                [sess.run(["water_content"], chunk_feed)[0]
                 for sess in ENSEMBLE]
            )  # (20, chunk, n_wp)

            all_mean.append(np.mean(preds, axis=0))
            all_std.append(np.std(preds, axis=0))
            all_q025.append(np.percentile(preds, 2.5, axis=0))
            all_q975.append(np.percentile(preds, 97.5, axis=0))

            if attn_available:
                # Collect property attention per ensemble member
                prop_attn = np.array([
                    sess.run(["property_attention_weights"],
                             chunk_feed)[0]
                    for sess in ENSEMBLE
                ])  # (20, chunk, H, 4, 4)
                # Average over ensemble, heads, then column mean
                avg = np.mean(np.mean(prop_attn, axis=0), axis=1)  # (chunk, 4, 4)
                col_means = np.mean(avg, axis=1)  # (chunk, 4)
                all_attn.append(col_means)

            progress.progress(
                (ci_idx + 1) / n_chunks,
                text=f"Processed {e:,}/{n_soils:,} soils")

        progress.empty()

        # Concatenate results — each is (n_soils, n_wp)
        mean_all = np.concatenate(all_mean, axis=0)
        std_all  = np.concatenate(all_std,  axis=0)
        q025_all = np.concatenate(all_q025, axis=0)
        q975_all = np.concatenate(all_q975, axis=0)

        # Soil IDs
        if "soil_id" in cols_lower:
            soil_ids = df[cols_lower["soil_id"]].values
        else:
            soil_ids = np.arange(1, n_soils + 1)

        # Stage per soil
        stage_map = {
            (1, 0, 0, 0): "Stage 0", (1, 1, 0, 0): "Stage 1",
            (1, 1, 1, 0): "Stage 2", (1, 1, 1, 1): "Stage 3",
        }
        stages = [stage_map.get(tuple(int(x) for x in mask[i]), "Custom")
                  for i in range(n_soils)]

        # Build output (long format: n_soils × n_wp rows)
        out = {"soil_id": np.repeat(soil_ids, n_wp)}

        for col in ["sand", "silt", "clay"]:
            out[col] = np.repeat(
                pd.to_numeric(df[cols_lower[col]], errors="coerce").values,
                n_wp)
        for col in ["bd", "oc", "ksat"]:
            if col in cols_lower:
                out[col] = np.repeat(
                    pd.to_numeric(df[cols_lower[col]],
                                  errors="coerce").values, n_wp)

        # Pass through any extra columns from the input CSV
        known = {"sand", "silt", "clay", "bd", "oc", "ksat", "soil_id"}
        for orig_col in df.columns:
            if orig_col.strip().lower() not in known:
                out[orig_col] = np.repeat(df[orig_col].values, n_wp)

        out["stage"] = np.repeat(stages, n_wp)
        out["water_potential_kPa"] = np.tile(wp_kpa, n_soils)
        out["theta_mean"] = mean_all.ravel()
        out["theta_std"]  = std_all.ravel()
        out["theta_q025"] = q025_all.ravel()
        out["theta_q975"] = q975_all.ravel()

        if attn_available and all_attn:
            attn_all = np.concatenate(all_attn, axis=0)  # (n_soils, 4)
            prop_labels = ["attn_texture", "attn_bd", "attn_oc", "attn_ksat"]
            for j, lbl in enumerate(prop_labels):
                out[lbl] = np.repeat(attn_all[:, j], n_wp)

        result_df = pd.DataFrame(out)

        # Summary
        stage_counts = pd.Series(stages).value_counts().sort_index()
        stage_str = ", ".join(f"{v:,} {k}" for k, v in stage_counts.items())
        st.success(
            f"**{n_soils:,} soils × {n_wp} water potential(s) "
            f"= {len(result_df):,} predictions.**  "
            f"Stages: {stage_str}."
        )

        # Preview
        st.dataframe(result_df.head(50), use_container_width=True,
                     hide_index=True)
        if len(result_df) > 50:
            st.caption(f"Showing first 50 of {len(result_df):,} rows.")

        # Download
        csv_buf = io.StringIO()
        result_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download results (CSV)",
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
from basic soil properties.  It uses a transformer-based architecture with
property-specific encoders, cross-attention layers that learn interactions
between properties, a monotonic output layer enforcing physically correct
behavior, and hierarchical training so one model handles any combination
of inputs.

Provide whatever properties you have — the model adapts automatically.
""")

    ab_col1, ab_col2 = st.columns(2)
    ab_col1.markdown(
        '[Model weights (HuggingFace)](https://huggingface.co/Teamrat/habit)'
        '&ensp;·&ensp;`pip install habit-ptf`')
    ab_col2.code("""from habit_ptf import load_ensemble
predictor = load_ensemble()
result = predictor.predict(soil_dataframe)""", language="python")

    st.markdown("#### How attention works in HABIT")
    st.markdown("""
The single-soil prediction panel shows three types of attention weights
extracted from the model.  Together they reveal *why* the model made a
particular prediction, not just *what* it predicted.

**Property attention** (shown by default) is a 4×4 self-attention matrix
over the four property embeddings (texture, bulk density, organic carbon,
Ksat).  We display the column-wise mean: the average attention each property
*receives* from all other properties.  Higher weight means the model relies
more on that property for the current soil.  Properties you did not provide
are masked and shown in gray — the model learns to ignore them.  Error bars
show the spread across the 20 ensemble members.

**Cross-attention** (inside the expander) captures pairwise interactions
between specific property pairs.  HABIT has two cross-attention modules:
texture↔bulk density (4 heads) and texture↔organic carbon (2 heads).
The heatmaps show how much each property in a pair attends to the other
after averaging over heads and ensemble members.  Strong off-diagonal
values indicate that the model found an informative interaction between
those properties for this particular soil.

**Water potential attention** (inside the expander) shows how the soil
embedding attends to different points along the water potential axis.
This reveals which part of the retention curve is most influenced by the
soil's properties — for example, coarse soils often show attention
concentrated at the wet end where drainage is rapid, while fine-textured
soils spread attention more evenly.
""")

    st.markdown("#### Ensemble spread")
    st.markdown("""
The shaded band and σ values represent the spread among 20 independently
trained models.  This is *not* a calibrated uncertainty interval.  However,
independent validation on 66,869 KSSL samples showed that ensemble spread
is monotonically associated with prediction error and can serve as an
indicator of prediction reliability — and of input data quality.
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
