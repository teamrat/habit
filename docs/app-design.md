# soil-habit.streamlit.app — design document

**Status:** design, not yet implemented. Supersedes conversational planning.
**Date:** 2026-08-19
**Applies to:** the Streamlit app in `HABIT-distribution/streamlit-app/`, and by
extension the `habit-ptf` package and the HuggingFace repo where noted.

This document exists so that no part of the design has to be recovered from
memory. Anything stated here as a fact is sourced. Anything not yet measured is
marked as such.

---

## 1. Model versions

Two models are distributed. Both are first-class; neither is deprecated.

| | **HABIT v1.0 (WRR 2026)** | **HABIT v1.1 (fixed-grid)** |
|---|---|---|
| Published in | Ghezzehei (2026), *WRR* 62, e2025WR042833 | technical note (in preparation) |
| Query behaviour | θ(ψ) depends on the full set of ψ requested | θ(ψ) depends only on ψ |
| ψ at inference | model input | not a model input |
| Role in the app | reproduction of the published results | everyday prediction, default |

**Naming rule.** User-facing text says *HABIT v1.0 (WRR 2026)* and
*HABIT v1.1 (fixed-grid)*. The word "legacy" is not used anywhere: v1.0 is the
peer-reviewed model and the app was published alongside the paper. Internally
the code may use `baseline` / `fixedgrid`, matching `HABIT-fixedgrid`.

**Why 1.1 and not 2.0.** The network is unchanged: `FixedGridHABIT` subclasses
`HABIT` and overrides only `call`; variable structure is identical and
checkpoints interchange in both directions. Same architecture, same training
data, same hyperparameters — only the evaluation protocol differs. A major bump
is reserved for an architectural change, such as redesigning the attention
arrangement. The ONNX artefact signature *does* change (§4.1); that break is
handled by the version selector and by `GRID_ID`, not by the version number.
Model version and package version are separate number lines — `habit-ptf` keeps
its own semver and its 0.2.x numbering is unrelated to the model version.

### 1.1 Why the fixed-grid version exists

In v1.0 the water-potential attention block uses a single query (the soil
embedding) against keys and values built from the requested ψ set, producing a
convex combination that is then repeated across every output point. The
consequence is that the predicted θ at 33 kPa changes depending on which *other*
potentials were requested in the same call. The dependence hierarchy is
extent ≫ density ≫ count, with count exactly inert (softmax normalisation makes
it so). Measured effect on a KSSL validation run: a single-point query at 33 kPa
had zero bias, while a 0.01–15000 kPa grid gave −0.015 to −0.028 cm³/cm³.

v1.1 removes the dependence structurally: `FixedGridHABIT.call` discards the
requested ψ before the network sees it, evaluates on a canonical grid, and
interpolates. Grid independence is therefore exact (`0.000e+00`) and is a
property of the architecture, not of the trained weights.

### 1.2 Canonical grid (part of the model definition)

```
log10(psi_kPa) = numpy.linspace(-2.0, 5.5, 151)     # 151 points, h = 0.05 decades
psi in [1.0e-2, 3.1623e5] kPa
GRID_ID = "logpsi_-2.0_5.5_151"
```

The endpoints are the clip bounds already enforced in training at
`preprocessing/data_preparation_full.py:211` (`np.clip(np.log10(psi), -2, 5.5)`).
The grid therefore interpolates and never extrapolates, by construction, at both
training and prediction time. No caller request can fall outside it.

**A checkpoint paired with a different grid is a different model and will not
announce itself.** `GRID_ID` must travel with every distributed artefact.

### 1.3 Evidence that the fixed grid costs nothing

From `HABIT-fixedgrid/PROGRESS_REPORT.md` §5, **pilot, n = 1** (seed 8028 =
WRR member 1), validation ΔRMSE (fixed-grid minus baseline):

| stage | ΔRMSE | ΔR² | ΔMVR |
|---|---|---|---|
| 0 (texture) | +0.00075 | −0.00548 | −1.03 pp |
| 1 (+BD) | +0.00002 | −0.00009 | −0.52 pp |
| 2 (+OC) | −0.00022 | +0.00116 | −0.39 pp |

Within 0.5 sd of seed-to-seed spread; monotonicity violations lower at every
stage. **This is one member and is not yet a result.** The 20-member ensemble is
training. No app copy should quote these numbers until the ensemble result is
written up.

---

## 2. Information architecture

```
[ Single Soil ]  [ Batch (CSV) ]  [ HABIT v1.0 (WRR 2026) ]  [ Help & FAQ ]  [ About ]
```

Five tabs, replacing the current three (`Single Soil`, `Batch (CSV)`, `About`).

| tab | model | contents |
|---|---|---|
| Single Soil | v1.1 | interactive prediction, plot, attention panels |
| Batch (CSV) | v1.1 | upload, fixed schema, long-form output |
| HABIT v1.0 (WRR 2026) | v1.0 | the current app's two modes, frozen |
| Help & FAQ | — | how to use, input schema, units, interpretation, troubleshooting |
| About | — | what HABIT is, versions, performance, citation, contact, **changelog** |

**Why a separate v1.0 tab rather than a global model switch.** The ψ controls
mean different things in the two models — model inputs under v1.0, display
controls under v1.1. A sidebar radio would silently change what the same three
number boxes do. Tabs let each control set carry exactly one meaning and one
label.

**Why the v1.0 tab is low-risk.** The current app *is* v1.0. The tab is a code
move, not a rewrite, and reproducibility requires that path stay behaviourally
identical to what the paper points at.

The v1.0 tab opens with a short standing note: this reproduces the published
model, θ depends on the set of potentials requested, and v1.1 is recommended for
new work.

---

## 3. Shared conventions

### 3.1 Units — fixed, never auto-detected

| property | unit |
|---|---|
| sand, silt, clay | percent by mass, summing to ~100 |
| bulk density | g/cm³ |
| organic carbon | percent by mass |
| Ksat | cm/day |
| water potential | kPa, magnitude (positive) |

Values that do not match are an **error**, not a reinterpretation. Texture that
does not sum to ~100 is rejected rather than treated as fractions. This rule is
absolute and applies to both model versions and to `habit-ptf`.

### 3.2 Preprocessing — identical for v1.0 and v1.1

```python
texture:  percent -> renormalise to fractions summing to 1 -> RobustScaler
bd:       raw g/cm3                                        -> RobustScaler
oc:       log(1 + 10 * OC_percent) / log(11)               -> RobustScaler
ksat:     log10(max(x, 1e-6))                              -> RobustScaler
```

Full-precision scaler parameters, verbatim from
`HABIT-training/data/processed/scaler_params.json`:

```python
SCALER_PARAMS = {
    "texture": {"center": [0.2712, 0.413, 0.172],
                "scale":  [0.45599999999999996, 0.454347652347652,
                           0.18299999999999997]},
    "bd":      {"center": [1.4],                "scale": [0.31000000000000005]},
    "oc":      {"center": [1.28],               "scale": [1.990208817]},
    "ksat":    {"center": [2.120606831056773],  "scale": [1.5132958130817589]},
}
```

Do not round these — the previous 4-decimal values shifted scaled inputs by up
to 1.2e-4.

Because preprocessing is shared, `verify_preprocessing.py` needs no change and
covers both versions. It reproduces the archived training tensors for all four
stages against `HABIT-WRR-dryad/data/processed`, using numpy alone.

### 3.3 Stage inference

Stage is determined per row from which properties carry values:

| stage | properties |
|---|---|
| 0 | texture only |
| 1 | + bulk density |
| 2 | + organic carbon |
| 3 | + Ksat |

Missing properties are fed as zeros with the corresponding mask column set to 0.
Rows in one batch may be at different stages. Texture is always required.

**Any combination of properties is valid input.** "Stage" is a name for the
training curriculum, not a constraint at inference. The model gates each
optional property independently — `habit.py` uses `properties_mask[:, 2]` for
organic carbon and `properties_mask[:, 3]` for Ksat, each with its own fallback
to un-enhanced features, and neither references bulk density. Texture + OC with
no bulk density is therefore a well-formed request, not an out-of-distribution
one. Nothing needs to be rejected or warned about.

`stage` is reported as an integer `0..3`, meaning the highest nested level
fully satisfied. A soil with texture + OC but no bulk density is stage 0 by that
rule even though organic carbon was used. Nothing is lost: the output echoes
every predictor column, so a blank `bd` beside a filled `oc` states exactly what
was supplied. `stage` is a summary; the echoed cells are the record.

### 3.4 Theme

Palette is defined once in `PALETTE` and drives both the CSS custom properties
(`--habit-*`) and the matplotlib figure colours, resolved from
`st.context.theme.type` with `st.get_option("theme.base")` as fallback. Do not
use `var(--text-color)`; it does not exist in Streamlit and fails silently.

---

## 4. Inference layer (v1.1)

### 4.1 ONNX signature

The exported v1.1 graph takes **no water-potential input**.

```
inputs:  texture (n,3)  bd (n,1)  oc (n,1)  ksat (n,1)  mask (n,4)
outputs: water_content                (n, 151)   # canonical grid, always
         property_attention_weights   (n, H, 4, 4)
         cross_attention_texture_bd   (n, H, 2, 2)
         cross_attention_texture_oc   (n, H, 2, 2)
         wp_attention_weights         (n, H, 1, 151)
```

The four attention tensors must be exported, not just `water_content` — the
single-soil panels and the batch attention checkbox both depend on them, and
v1.0 already exports all five under these names. `wp_attention_weights` is now
over the canonical grid rather than over a user-chosen set, which is what makes
the panel a statement about the soil rather than about the request (§5).

This differs from v1.0, whose graph takes `water_potential` and returns
attention tensors alongside `water_content`. The signature difference makes the
two artefacts structurally impossible to confuse — a direct answer to the hazard
that fixed-grid weights load silently into a baseline class and predict the old
way, since variable structure is identical in both directions.

Retrieve outputs **by name**. ONNX orders outputs alphabetically, and
`sess.run(None, feed)[0]` returns `cross_attention_texture_bd`, not water
content. This has already caused one bug.

### 4.2 Prediction path

1. Build the scaled feed for the unique soils.
2. Run all 20 members → 20 curves of 151 points per soil.
3. Interpolate **each member's** curve, linearly in log₁₀ψ, to the requested
   output points.
4. Aggregate across members: mean, sd, 2.5% and 97.5% quantiles.

Order matters for σ but not for the mean: linear interpolation commutes with
averaging, so the mean is identical either way, while sd is not. Interpolating
per member first is the order that matches what a per-point query would give.
Cost is negligible.

### 4.3 Consequences

- **One inference per unique soil**, regardless of how many output points are
  requested. Batch cost stops scaling with the ψ list.
- The 151-point curve is cacheable per soil, so re-plotting at a different range
  or resolution is free.
- Interpolation is visible numpy in the app, auditable, not frozen in a graph.

There is no separate "interpolation error" to report. The interpolation is
inside the model definition: `FixedGridHABIT.call` interpolates during
*training* as well, so the loss compares interpolated values at each sample's
measurement points to the observations, and the network fitted its 151 nodes
under that constraint. Any interpolation penalty is already inside the reported
RMSE, and there is no continuous ground-truth curve to measure against. The only
thing a finer grid would test is a different model, which would need retraining.

### 4.4 Domain guard

ψ outside [1.0e-2, 3.1623e5] kPa is an **error**, not a clamp. Silent clamping
would return a number that looks valid while being conditional on something the
user never saw.

---

## 5. Tab: Single Soil (v1.1)

Layout unchanged: inputs left (ratio 1), results right (ratio 2.5).

**Inputs.** Texture three-up with a running sum indicator; toggles plus number
inputs for BD / OC / Ksat. Number-input labels centred above their boxes.

**What changes.** The `Min (kPa)` / `Max (kPa)` / `Points` controls are
**display controls only**. They set the plot range and resolution; they are not
sent to the model and cannot change any predicted value. Relabel the group
accordingly — *Plot range*, not *Water potentials* — and say so in one line of
help text.

**Summary cards.** θ_sat (0.01 kPa), FC (33 kPa), PWP (1500 kPa). Under v1.1
these are **always defined**, because the canonical grid always spans them. The
range-coverage check and the em-dash blanking branch are both deleted. Values
come from the canonical curve by log-ψ interpolation, not from the displayed
grid.

**Table.** Reports the ψ actually evaluated, formatted from the value, not from
a standard label.

**Attention panels.** Property attention (4×4 self-attention, column-wise mean,
error bars across members), cross-attention heatmaps, water-potential attention.

Water-potential attention is kept, with its axis relabelled as the canonical
grid. Under v1.1 the tensor is `(H, 1, 151)` and no longer depends on what the
user asked for, which makes it a property of the soil rather than of the
request — a cleaner reading than v1.0's, not a degraded one. The displayed range
follows the plot range; the underlying tensor does not change with it.

---

## 6. Tab: Batch (CSV) — v1.1

### 6.1 Input schema — exact column names, nothing else accepted

| column | required | unit | notes |
|---|---|---|---|
| `soil_id` | **yes** | — | unique per soil in wide form; repeated in long form |
| `sand` | **yes, with values** | % | |
| `silt` | **yes, with values** | % | |
| `clay` | **yes, with values** | % | |
| `bd` | header required | g/cm³ | blank if unavailable |
| `oc` | header required | % | blank if unavailable |
| `ksat` | header required | cm/day | blank if unavailable |
| `psi_kpa` | optional | kPa | **its presence selects long form** |
| `theta_obs` | optional | cm³/cm³ | observed water content, passed through |
| `note` | optional | — | free text, passed through |

Rules:

- The header must always carry every predictor column. Blank cells mark what is
  missing; stage is inferred per row from which of `bd` / `oc` / `ksat` carry
  values.
- **Texture must carry values.** Blank texture is an error — there is no
  prediction below stage 0.
- Any column name outside this set is an **error**. There is no catch-all
  pass-through: a mistyped `clay_pct` must fail loudly rather than sail through
  and silently drop the soil to a lower stage.
- **`theta_obs` without `psi_kpa` is an error.** An observed water content with
  no stated potential cannot be placed on the curve; repeating it against every
  output point would look like a valid comparison and be meaningless.
- `note` has no such constraint and repeats freely across a soil's rows.

### 6.2 Two input forms

**Wide** — no `psi_kpa` column, one row per soil. Output points come from the
UI, either:

- a common ψ array (text box, default the 19-value list
  `0.01, 0.1, 0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 33, 50, 70, 100, 300, 1500, 15000`), or
- the full canonical 151-point grid.

A repeated `soil_id` in wide form is an error.

**Long** — `psi_kpa` present, multiple rows per soil, each with its own
potential. This is the measurement-comparison form.

Pipeline:

1. Group by `soil_id`.
2. **Validate that every row sharing a `soil_id` carries identical predictors.**
   A conflict is an error naming the ID and the disagreeing column — never
   first-wins. This is the one place the long form can go quietly wrong.
3. One prediction per unique soil on the canonical 151 points.
4. Interpolate each member at that row's ψ; aggregate.
5. Return input rows in original order with prediction columns appended.

`psi_kpa` present but `soil_id` missing → error, not a fallback to treating every
row as its own soil. A repeated `(soil_id, psi_kpa)` pair is allowed and returns
identical values.

### 6.3 Output — always long form

One row per soil × ψ, regardless of input form or how few potentials were asked
for.

```
soil_id, sand, silt, clay, bd, oc, ksat,
psi_kpa, theta, theta_sd, theta_q025, theta_q975, stage
        [, theta_obs] [, note] [, attn_texture, attn_bd, attn_oc, attn_ksat]
```

- Predictors are echoed back as supplied, blank where missing. This matches what
  the v1.0 batch tab already does; dropping it would be a gratuitous difference.
- Column names differ from the v1.0 tab's output (`psi_kpa` vs
  `water_potential_kPa`, `theta` vs `theta_mean`, `theta_sd` vs `theta_std`).
  That difference is deliberate — see §7 — and is documented in Help & FAQ.
- Long-form input: input row order preserved.
- Wide-form input: soils in input order, ψ ascending within each soil.
- `theta_obs` and `note` appear only if supplied. In wide form `note` repeats
  across every ψ row for that soil.
- Attention columns only if the checkbox is set.

### 6.4 Cost

Per unique soil, not per row. A 10,000-row measurement file covering 200 soils
costs 200 predictions × 20 members. Under v1.0 the same file cost 10,000
predictions *and* each soil's answer depended on which potentials happened to sit
in its rows — which is exactly what made measurement comparison unreliable.

### 6.5 Templates

Two downloadable templates, differing only in the two optional columns and in
whether soils repeat. Shared columns keep the same order in both.

`habit_template_wide.csv`

```
soil_id,sand,silt,clay,bd,oc,ksat,note
sample_1,40,40,20,1.35,1.2,50,
sample_2,65,25,10,1.50,,,
sample_3,90,5,5,,,,coarse
```

`habit_template_long.csv`

```
soil_id,sand,silt,clay,bd,oc,ksat,psi_kpa,theta_obs,note
sample_1,40,40,20,1.35,1.2,50,10,0.312,
sample_1,40,40,20,1.35,1.2,50,33,0.268,
sample_1,40,40,20,1.35,1.2,50,1500,0.141,
sample_2,65,25,10,1.50,,,33,0.191,
sample_2,65,25,10,1.50,,,1500,0.087,
```

*The `theta_obs` values above are illustrative placeholders, not measurements.*

Between them the templates demonstrate: a blank cell for an unavailable
property, the optional `note`, and — in the long form — that predictors must
repeat identically on every row of a soil.

---

## 7. Tab: HABIT v1.0 (WRR 2026)

Frozen reproduction of the published model. Contains the current app's single-
soil and batch modes, selected by a radio inside the tab.

- Behaviour is **not** modernised. The ψ controls remain model inputs. Summary
  cards remain blank outside the requested range.
- The one thing carried over from the fixes: preprocessing. v1.0 as shipped in
  the app before 2026-08-19 had the wrong OC transform; the corrected transform
  is what the paper's own analysis scripts used, so applying it makes the tab
  match the paper rather than diverge from it.
- Standing note at the top of the tab: what the query dependence is, that it is
  documented in the technical note, and that v1.1 is recommended for new work.
- Batch mode here keeps the v1.0 CSV conventions. It does **not** adopt the
  fixed schema of §6 — changing it would change what the published path does.
  Specifically it keeps: optional `soil_id` (auto-numbered when absent), the
  catch-all pass-through of any unrecognised column, and the output column names
  `water_potential_kPa` / `theta_mean` / `theta_std`, and its own template CSV
  exactly as it stands today. The catch-all is precisely
  the failure mode §6.1 forbids — a mistyped `clay_pct` passes through silently
  and the soil drops a stage. It stays in v1.0 because removing it would change
  behaviour; it is one of the reasons v1.1 is the recommended path.

---

## 8. Tab: Help & FAQ

Everything operational, moved out of About.

1. **Units** — the table from §3.1, stated as absolute.
2. **How to read the output** — mean, ensemble band, what σ is and is not.
3. **Batch CSV** — the full §6 schema, both forms, downloadable templates
   (wide and long).
4. **What the stages mean** — §3.3, and that rows may differ.
5. **Which version should I use?** — v1.1 unless reproducing the paper.
6. **Why does v1.0 give a different answer than v1.1?** — the query dependence,
   short version, pointing at the technical note.
7. **Attention panels** — the explanatory text currently in About §"How
   attention works in HABIT", including the bulk-density-slot explanation.
8. **Troubleshooting** — texture sum rejected, unknown column, `theta_obs`
   without `psi_kpa`, ψ out of domain, duplicate `soil_id`.
9. **Install locally** — `pip install habit-ptf`, the two-line example,
   `model_version=` selection.

---

## 9. Tab: About — brief

Kept short. Everything how-to lives in Help & FAQ.

1. **What HABIT is** — one paragraph, the architecture summary already written.
2. **Versions** — the §1 table, two or three sentences.
3. **Performance** — the independent-test table (R², RMSE, MAE with cluster
   bootstrap CIs) for v1.0. v1.1 numbers added only when the ensemble result is
   written.
4. **Citation** — Ghezzehei (2026), *WRR* 62, e2025WR042833,
   doi:10.1029/2025WR042833; technical note added when published.
5. **Contact** — "Questions: [UC Merced Soil Physics
   Lab](https://soilphysics.ucmerced.edu)". **No email address anywhere in the
   app**, including the footer and any template files.
6. **License** — MIT (code and weights); training data CC BY 4.0.
7. **Changelog** — §10.

---

## 10. Changelog (lives in About)

Release notes, not a commit log. One entry per model version, newest first.
Rendered from a single source of truth — a `CHANGELOG` constant or a markdown
file read at startup — so it cannot drift from the code.

**v1.1 — fixed-grid (pending; publishes with the technical note)**
Predictions are evaluated on a fixed canonical grid and interpolated to the
potentials you ask for, so a value at 33 kPa no longer depends on which other
potentials were requested alongside it. v1.0 remains available on its own tab.

**v1.0 — original release**
The model published in Ghezzehei (2026), *Water Resources Research*, 62,
e2025WR042833. App preprocessing was corrected in August 2026; predictions
involving organic carbon changed accordingly.

---

## 11. Resources and deployment

**Platform limit.** Streamlit Community Cloud gives 690 MB minimum, 2.7 GB
maximum per app. Design to the floor.

**Ensemble size.** 7.4 MB per member on disk (148 MB per ensemble), but 12.0 MB
per member **resident** — 240 MB per loaded ensemble. Documentation saying
"~50 MB" or "~2.5 MB each" is wrong in both senses and needs correcting in
`streamlit-app/README.md`, `habit-ptf/README.md` (three places) and
`habit-ptf/MODEL_WEIGHTS.md`.

**`enable_cpu_mem_arena = False` is mandatory.** ONNX Runtime's CPU arena is on
by default, grows to the high-water mark of any single inference call, and never
returns memory for the life of the session. Measured with the configuration that
ships today (`CHUNK = 500`, arena default):

| batch | RSS |
|---|---:|
| 20 sessions loaded, no inference | 284 MB |
| one 500-soil chunk × 19 potentials (today's default) | **888 MB** |
| same, user enters 50 potentials | **2,541 MB** |
| same, v1.1 canonical grid of 151 points | **5,060 MB** |

The first row of real work already exceeds the 690 MB floor. This is a property
of the app as deployed, not a consequence of adding a second model. With the
arena disabled, a 500-soil × 151-point batch costs 41 MB of transient allocation
instead of 4.5 GB:

| state | RSS | peak |
|---|---:|---:|
| one ensemble resident | 276 MB | 277 MB |
| + 500-soil × 151 batch | 278 MB | 544 MB |
| both ensembles resident | 511 MB | 544 MB |
| both resident, 100-soil × 151 on each | 503 MB | 560 MB |

Throughput was measured, not assumed: with the arena off a 100-soil x 151-point
ensemble pass takes 2,165 ms against 3,229 ms with it on (0.67x), and the
single-soil interactive path is 52.9 ms against 55.1 ms (0.96x, within noise).
Disabling the arena is faster here, not slower — a multi-GB arena on a 2-core
box costs first-touch page faults and cache locality. Weights are unaffected:
they are loaded once at `InferenceSession` construction and live for the
session's lifetime. The arena holds only intermediate tensors, so nothing is
reloaded per chunk and numerics are unchanged.

**Chunk size.** `CHUNK = 100`, not the current 500. With the arena off this
matters much less, but it also caps the numpy-side `(20, chunk, n_wp)` array.

**Memory strategy.** One cached loader keyed by version with
`st.cache_resource(max_entries=1)`, plus an explicit
`ctypes.CDLL("libc.so.6").malloc_trim(0)` immediately after obtaining an
ensemble. Eviction alone is not enough: `del` + `gc.collect()` returns memory to
the allocator but glibc keeps the pages, and RSS is what gets a container
killed — measured 4,327 MB → 704 MB on eviction → 292 MB after the trim.
Trimming after retrieval rather than inside the loader makes the result
independent of Streamlit's eviction ordering.

Both ensembles resident is *possible* at 511 MB, but leaves only ~60–90 MB of
margin under the floor once Streamlit's own runtime is counted. One at a time is
~276 MB and comfortable.

Weights cache locally under `~/.cache/habit-ptf/onnx`, so a version switch
reloads from disk, not from the network.

**No row ceiling.** Cost is bounded by unique soils, not rows. Twenty members ×
151 points × 4 bytes is 12 KB of raw curve per soil, and the existing chunking
reduces each chunk to mean/sd/quantiles before the next is run, so peak memory
does not grow with the file. Streamlit's own `server.maxUploadSize` (200 MB by
default) is already a de facto ceiling on input. Wall-clock throughput has not
been measured; if it turns out to be the binding constraint, the answer is a
progress indicator and a stated expectation, not a hard cap.

**HuggingFace layout.**

```
Teamrat/habit
  v1.0/   20 ONNX members            (existing files, moved or aliased)
  v1.1/   20 ONNX members + grid.json {"grid_id": "logpsi_-2.0_5.5_151", ...}
  README.md  model card documenting both versions and the grid
```

v1.0 stays downloadable indefinitely — the app and the paper both point at it.

**`habit-ptf`.** Gains `model_version=`, defaulting to `"v1.1"`, with `"v1.0"`
documented as the reproduction path. On load it asserts that the artefact's
`GRID_ID` matches its compiled-in canonical grid and refuses a mismatch.

**The `GRID_ID` requirement is the item most likely to be lost between here and
distribution.** It must land in three places: the HF model card, `grid.json`
beside the v1.1 weights, and an assertion in `habit-ptf`.

---

## 12. Validation

### 12.1 Input errors

Every one of these is an error that stops the run with a named cause. None is a
silent correction.

| condition | message names |
|---|---|
| texture sum not ≈100 | the row(s) and the observed sum |
| blank texture | the row(s) |
| unknown column | the column name and the accepted set |
| missing predictor column header | which one |
| `theta_obs` without `psi_kpa` | both column names |
| `psi_kpa` without `soil_id` | both column names |
| duplicate `soil_id` in wide form | the id |
| conflicting predictors for one `soil_id` | the id and the disagreeing column |
| ψ outside [1e-2, 3.1623e5] kPa | the offending value and the domain |
| unparseable ψ list | the offending token |

### 12.2 Test coverage

`verify_preprocessing.py` covers the transforms and nothing else. The v1.1 work
adds paths that no test touches. Minimum coverage before release:

| target | test |
|---|---|
| interpolation | linear function reproduced exactly; grid nodes exact; agreement with `numpy.interp` |
| aggregation order | mean identical whether interpolation precedes or follows averaging; σ documented as order-dependent |
| grid independence, end to end | same soil, three different requested ψ sets → bitwise-identical θ at shared points |
| CSV schema | one case per row of §12.1, asserting the error is raised and names the cause |
| long-form round trip | row order preserved; `theta_obs` and `note` unmodified |
| dedup | a long file whose soils are shuffled gives the same answers as the sorted version |
| v1.0 regression | the frozen tab reproduces stored reference output byte for byte |

The last one is the important one: it is what makes "v1.0 is frozen" a checkable
claim rather than an intention.

---

## 13. Sequencing

| step | blocked on |
|---|---|
| Move current app into the v1.0 tab; add Help & FAQ; shorten About; add changelog | nothing |
| Shared: fixed CSV schema parser, long-form writer, validation messages | nothing |
| ONNX re-export of v1.1 from `FixedGridHABIT` with the 151-point signature | 20-member ensemble |
| v1.1 Single Soil and Batch tabs | the export |
| Flip default to v1.1; publish changelog entry | technical note published |

No discovery risk sits in front of any of this. Grid independence is structural
and already verified; the remaining `HABIT-fixedgrid` items are evaluation and
write-up.

---

## 14. Out of scope

- Retraining or architecture changes.
- Showing both models' predictions side by side in one view. An interested user
  can do that through `habit-ptf`; it is not an everyday-user feature.
- Per-row output-point columns beyond `psi_kpa` (e.g. multiple pass-through
  columns). One `note` and one `theta_obs` cover the stated use cases.
- Uploading measured curves for automatic goodness-of-fit statistics. The long
  form plus `theta_obs` gives the user everything needed to compute their own.

---

## 15. Open items

None. Memory behaviour was measured on 2026-08-19 (§11); inference throughput
with `enable_cpu_mem_arena = False` is the one quantity still unmeasured, and it
affects tuning, not design.
