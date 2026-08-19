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

All colours live in ONE place: the `PALETTE` dict near the top of `app.py`,
with a `light` and a `dark` entry sharing the same keys. Do not hardcode a hex
value anywhere else.

`THEME = _active_theme()` resolves the live theme from `st.context.theme.type`,
falling back to `theme.base` in config.toml when that is unset (per Streamlit's
docs it can be unset on a session's first script run and briefly stale right
after a theme switch). `C = PALETTE[THEME]` is then used two ways:

1. emitted as `--habit-*` CSS custom properties on `:root`, which the whole
   stylesheet references — these are OUR variables, so they always resolve
2. assigned to the `PLOT_*` constants, so the matplotlib figures follow the
   page theme instead of always rendering warm-paper on a dark background

Only bg / secondary / text / border / accent in the dark entry come from
`[theme.dark]` in config.toml; surface, muted, faint, accent_light, accent2 and
ghost were derived by inverting lightness and checked for WCAG contrast
(lowest pair 4.75:1 against a 4.5 threshold).

**Never use `var(--text-color)`, `var(--border-color)` or `var(--primary-color)`.**
Streamlit does not define them. They fail silently to whatever literal fallback
is written after the comma, which matched the light theme by luck and left the
title near-invisible in dark mode for several releases.

| Key | Light | Dark | Use |
|-----|-------|------|-----|
| bg | `#F1ECE3` | `#1B1815` | page + plot background |
| surface | `#FBF8F2` | `#221E1A` | cards, callouts, legend |
| secondary | `#E7E0D2` | `#201C18` | sidebar, expanders |
| text | `#2A2621` | `#ECE5D9` | body text |
| muted | `#63594F` | `#B3A897` | axis labels, footer |
| faint | `#726657` | `#9A8E7D` | captions, card labels |
| border | `#DDD5C6` | `#38322A` | card borders, grid lines |
| accent | `#A24A28` | `#E39468` | primary, links, curves, eyebrows |
| accent_light | `#D4835E` | `#F0B492` | individual ensemble members |
| accent2 | `#4B6B54` | `#8FB89A` | OC cross-attention, texture-ok |
| accent2_deep | `#2D4032` | `#C7DCCD` | far end of the moss colormap |
| ghost | `#C8BCA8` | `#4A4238` | empty-state icon |

### Fonts:
- **Body**: Inter (via Google Fonts in config.toml)
- **Title**: Fraunces (optical size, serif)

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

## Predictions depend on the REQUESTED water-potential set

`wp_attention` uses the soil embedding as a single query and the requested psi
values as keys/values (habit.py 331-338). Its output — one summary vector of
the whole requested set — is added to the soil embedding (line 345), and that
combined vector is repeated across all points to generate the curve parameters
(347-356). So the curve is produced jointly and the requested grid conditions
it. This is intended: training and testing evaluated each sample at its own
measured psi points.

Measured, theta at 33 kPa, same soil, 5 members:

| grid | loam | sand | clay |
|---|---|---|---|
| standard 12 pts 1-15000 | 0.3499 | 0.1300 | 0.4208 |
| log 50 pts 0.1-15000 | 0.3309 | 0.1219 | 0.4215 |
| log 20 pts 10-1500 | 0.3544 | 0.1412 | 0.4029 |
| just [33] | 0.3178 | 0.1228 | 0.4013 |
| spread | 0.041 | 0.019 | 0.021 |

**Never add a psi point the user did not request.** The app sends exactly
`np.logspace(min, max, n_pts)`. theta_sat / FC / PWP are interpolated from that
grid when 0.01 / 33 / 1500 fall inside [min, max] inclusive, and shown as a
dash otherwise — never extrapolated, never obtained by padding the request.

An earlier version injected the three reference points so the cards could be
exact. That turned a min=max=33 request into a 3-point one, moved theta by
0.026, and made the app disagree with the habit-ptf package for the same soil.
Do not reintroduce it.

Repeated copies of one psi value are harmless (attention over N identical keys
gives the same context vector as over one), which is why min == max with
n_pts > 1 still matches a genuine single-point request.

To compare the app with the package, pass the same grid — not just the same
soil. That is the first thing to check if they ever disagree.

## Property attention: the BD asymmetry

At Stage 0 the property-attention bar chart shows a non-zero weight for bulk
density even though none was supplied. This is correct model behaviour, not a
display bug. In `habit/model/habit.py` (identical in HABIT-training and
HABIT-WRR-dryad, md5 54a9a256...):

- line 179 — `bd_features = self.bd_encoder(bd)` has NO mask gate, unlike
  lines 186-187 where `oc_features` and `ksat_features` are multiplied by
  their mask bit
- line 289 — the attention mask is built as
  `tf.concat([tf.ones_like(properties_mask[:, 0:2]), properties_mask[:, 2:4]])`,
  so texture and BD are hard-coded as always available; only OC and Ksat are
  actually masked out of the softmax
- line 224 vs 253 — when the availability bit is 0, `bd_enhanced_by_texture`
  falls back to `bd_features` (a non-zero learned vector), whereas
  `oc_enhanced_by_texture` falls back to `oc_features_safe`, already zeroed

So OC and Ksat are suppressed twice and read exactly 0.0000; BD is suppressed
zero times. The BD bar is the model's learned encoding of "bulk density
unknown", which is genuinely informative — flipping only the mask bit with the
BD tensor held at 0.0 moves theta by ~0.017 cm3/cm3 and the BD attention from
~0.34 to ~0.50 for a medium-textured soil. Do not "fix" this in the chart.
The About tab explains it to users.

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
