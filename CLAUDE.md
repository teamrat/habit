# CLAUDE.md — Streamlit App Maintenance Guide

## What this is

**soil HABIT** — a Streamlit web app for predicting soil water retention curves using a 20-member deep ensemble of ONNX models. Live at https://soil-habit.streamlit.app. Source repo: https://github.com/Teamrat/habit.

## Architecture

- **`app.py`** (~1100 lines) — single-file Streamlit app, everything lives here
- **`.streamlit/config.toml`** — theme config (Inter + Fraunces fonts, warm paper palette, light + dark themes)
- **`requirements.txt`** — `streamlit>=1.41`, `onnxruntime`, `numpy`, `pandas`, `huggingface_hub`, `matplotlib`
- **`figures/`** — PNG figures displayed in About tab (from WRR supplemental)
- **`upload_onnx_to_hf.py`** — one-time script to push ONNX weights to HuggingFace
- **`app_backup.py`** — pre-attention backup of app.py
- **`verify_preprocessing.py`** — checks preprocessing against the archived training tensors

## Deployment

Streamlit Community Cloud, auto-deploys from `main` branch on push. 1GB RAM limit.

```bash
# From the streamlit-app/ directory:
git add app.py .streamlit/config.toml
git commit -m "description"
git push
```

App rebuilds automatically. If it gets stuck loading, reboot from the Streamlit Cloud dashboard — do NOT assume code errors without checking.

## App structure (app.py)

### Sections in order:

1. **Imports + page config** (lines 1–25)
2. **CSS** (lines 27–108) — all custom styling in one `<style>` block
3. **Constants** — `PLOT_*` palette vars, `SCALER_PARAMS` for input normalization
4. **`load_ensemble()`** — downloads 20 ONNX models from HuggingFace `Teamrat/habit` repo, cached with `@st.cache_resource`
5. **`prepare_inputs()`** — builds ONNX feed dict for single soil (normalizes, creates mask, builds WP array)
6. **`prepare_batch_inputs()`** — same for batch CSV
7. **`run_ensemble_with_attention()`** — runs all 20 models, collects water content + 4 attention outputs
8. **`make_plot()`** — main water retention curve plot (matplotlib, log-scale x-axis)
9. **Header** — "soil HABIT" title in Fraunces, tagline below
10. **Tab 1: Single Soil** — left panel (inputs) + right panel (results)
11. **Tab 2: Batch CSV** — upload CSV, chunked inference, optional attention columns
12. **Tab 3: About** — performance table, ensemble spread figures, attention explanation, install section
13. **Footer** — single line with lab link, paper DOI, copyright

### Input panel (Single Soil tab, left column):
- Texture (sand/silt/clay) — always required, 3 columns
- Optional properties — BD, OC, Ksat with toggle + inline input
- Water potential range — min/max/points on one row
- +/- stepper buttons hidden via CSS

### ONNX model outputs (5 named outputs per model):
- `water_content` — predicted θ at each WP point
- `property_attention_weights` — (batch, 4 heads, 4, 4) self-attention over properties
- `cross_attention_texture_bd` — (batch, 4 heads, 2, 2) texture↔BD interaction
- `cross_attention_texture_oc` — (batch, 2 heads, 2, 2) texture↔OC interaction
- `wp_attention_weights` — (batch, 4 heads, 1, num_wp) attention along WP axis

### Attention visualizations (Single Soil only):
- **Property attention bar chart** — visible by default, column-mean of 4×4 attention, ensemble mean ± σ
- **Cross-attention heatmaps** — in expander, terracotta colormap (BD) and moss colormap (OC)
- **WP attention line plot** — in expander, shows where along the WP axis the model focuses

### Batch mode:
- Parses CSV, auto-detects stage per row based on available columns
- Chunked inference (500 soils per chunk) to stay within memory
- Requests outputs by name: `sess.run(["water_content"], feed)[0]` — critical because ONNX sorts outputs alphabetically
- Optional attention columns in download CSV

### Metric cards:
- θ_sat (0.01 kPa), FC (33 kPa), PWP (1500 kPa) — constants `WP_SAT`, `WP_FC`, `WP_PWP`
- These three are always appended to the user's grid via `np.union1d`, so the
  cards read **exact** predictions rather than interpolated or clamped values.
- `display_mask` (`np.isin(wp_all, wp_user)`) removes reference-only points
  again from the plot, the standard-WP table, the WP-attention chart and the
  CSV. A reference point the user asked for anyway stays visible.
- 0.01 kPa is the wet-end limit of training: preprocessing clamped log10(ψ) at
  −2.0, so 0 of 66,490 training points lie below it. Do not lower this without
  acknowledging it is extrapolation.

## Design language

Matches [Soil Physics Lab](https://soilphysics.ucmerced.edu) at UC Merced.

### Colors:
| Token | Light | Dark | Use |
|-------|-------|------|-----|
| Background | `#F1ECE3` | `#1B1815` | Page bg |
| Surface | `#FBF8F2` | — | Cards, legend bg |
| Secondary bg | `#E7E0D2` | `#201C18` | Sidebar, expanders |
| Text | `#2A2621` | `#ECE5D9` | Body text |
| Text muted | `#63594F` | — | Axis labels, secondary |
| Text faint | `#726657` | — | Tagline, captions |
| Border | `#DDD5C6` | `#38322A` | Card borders, grid lines |
| Accent (clay) | `#A24A28` | `#E39468` | Primary, links, curves |
| Accent light | `#D4835E` | — | Individual ensemble members |
| Accent2 (moss) | `#4B6B54` | — | OC cross-attention, texture-ok |

### Fonts:
- **Body**: Inter (via Google Fonts in config.toml)
- **Title**: Fraunces (optical size, serif)

### Plot palette (matplotlib):
All plots use `PLOT_*` constants defined near top of app.py. Background is warm paper (`#F1ECE3`), curves are terracotta (`#A24A28`), ensemble members are lighter (`#D4835E`).

### CSS conventions:
- `.section-label` — uppercase eyebrow labels in accent color with 0.14em letter-spacing
- `.metric-card` — surface bg, border, centered value + label
- `.ens-note` — left-bordered callout box
- `.app-footer` — border-top separator, centered, uses CSS variables for theme compatibility
- Title and subtitle use `var(--text-color)` for dark mode compatibility

## Input preprocessing — must match training exactly

Units are **fixed**. Nothing is auto-detected or converted: texture and OC in
percent, BD in g/cm³, Ksat in cm/day.

| Property | Transform | Then robust-scale by |
|----------|-----------|----------------------|
| Texture | renormalize sand/silt/clay to fractions summing to 1 | center `[0.2712, 0.413, 0.172]`, scale `[0.456, 0.454347652347652, 0.183]` |
| BD | none (raw g/cm³) | center `1.4`, scale `0.31` |
| OC | `log(1 + 10·OC%) / log(11)` | center `1.28`, scale `1.990208817` |
| Ksat | `log10(max(Ksat, 1e-6))` | center `2.120606831056773`, scale `1.5132958130817589` |

The OC transform is the one that bites. It is **not** `log1p`, and OC is **not**
converted to a fraction first. Source of truth: `transform_organic_carbon()` in
`HABIT-training/preprocessing/data_preparation_full.py`.

Verify any preprocessing change with `verify_preprocessing.py`, which
reproduces the archived training tensors in `HABIT-WRR-dryad/data/processed/`
(raw `*_original_data.json` → scaled `*_data.npz`) across all four levels and
both the single-soil and batch paths. Expect agreement to ~2e-7 — the float32
floor, since ONNX inputs are float32 and the archive is float64.

Do not round the scaler constants; copy them verbatim from
`HABIT-training/data/processed/scaler_params.json`. The 4-decimal values
previously in the app shifted scaled inputs by up to 1.2e-4.

The `habit-ptf` package (`HABIT-distribution/habit-ptf/`) had the same class
of OC bug and was fixed on 2026-08-19; it now uses the identical transforms
and its own `verify_preprocessing.py`. Keep the two in sync — if you change a
transform here, change it there.

## Key gotchas

1. **ONNX output ordering** — outputs are alphabetical, NOT in definition order. Always request by name: `sess.run(["water_content"], feed)`. Using `sess.run(None, feed)[0]` returns `cross_attention_texture_bd`, not water content.

2. **Streamlit Cloud Python version** — currently runs Python 3.14. Matplotlib mathtext can have issues; if log-scale tick labels crash, the fix is `FuncFormatter` (already reverted — a reboot fixed it last time).

3. **`use_container_width` deprecated** — Streamlit now wants `width="stretch"` or `width="content"`. Already migrated.

4. **Streamlit theming API (>=1.41)** — fonts load via `font = "FontName:css_url"` syntax in config.toml. No need for CSS @import hacks.

5. **Tab styling** — Streamlit's BaseWeb styles override custom CSS for tab buttons. Don't try to restyle tabs via CSS — it won't take effect.

6. **1GB RAM limit** — batch inference is chunked at 500 soils. Don't load all 20 models' full outputs at once for large batches.

## References

- **Paper**: Ghezzehei, T.A. (2026). *Water Resources Research*, 62, e2025WR042833. [doi:10.1029/2025WR042833](https://doi.org/10.1029/2025WR042833)
- **Model weights**: [huggingface.co/Teamrat/habit](https://huggingface.co/Teamrat/habit)
- **Lab**: [soilphysics.ucmerced.edu](https://soilphysics.ucmerced.edu)
- **Live app**: [soil-habit.streamlit.app](https://soil-habit.streamlit.app)

## User preferences

- Be concise and direct
- Don't guess — verify before claiming
- Don't run git — the user handles git commands themselves
- Don't change conda/TF environment
- R scripts: transparent data prep, no plotting functions, use ggarrange
