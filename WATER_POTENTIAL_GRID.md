# Choosing the water-potential grid

## Why the grid matters

HABIT predicts a retention curve as a single object, not point by point. The
water-potential attention layer takes the soil embedding as its query and the
**requested set of water potentials** as its keys and values; the resulting
summary is added to the soil embedding, and that combined vector generates the
curve parameters for every point (`habit/model/habit.py`, lines 331-356).

The requested grid is therefore part of the query, not just the sampling of an
answer. Two consequences:

1. A given θ(ψ) depends on the other potentials requested alongside it.
2. A grid whose *shape* is unlike the grids seen in training pushes the query
   out of distribution, which shows up as bias rather than noise.

## What the training grids looked like

Measured over the 5,453 training samples in
`HABIT-WRR-dryad/data/processed/*/train_original_data.json` (97,499 ψ values
in total). Each sample carries its own measured points, and the model was
trained and evaluated on exactly those.

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| points per sample | 4 | 6 | **10** | 23 | 241 |
| log10 ψ minimum | −2.00 | −2.00 | **−2.00** | −1.01 | 3.42 |
| log10 ψ maximum | 0.96 | 3.17 | **3.18** | 4.06 | 5.50 |
| span, decades | 1.23 | 4.27 | **5.18** | 5.18 | 7.50 |

So a wide span reaching down to 0.01 kPa is typical: the median sample starts
at exactly 0.01 kPa and spans 5.2 decades. **Range is not the problem.**

Note 0.01 kPa is a floor, not a measurement: preprocessing clamped log10(ψ) at
−2.0, and 3.2% of all training values sit exactly on that clamp. Nothing below
it was ever seen.

## Density is the problem

Where the training points actually fall, as a share of all 97,499 values:

| ψ band | share of training points |
|---|---|
| < 1 kPa | 14.4% |
| 1 – 10 | 32.5% |
| 10 – 100 | 32.7% |
| 100 – 1000 | 11.2% |
| > 1000 | 9.2% |

Two thirds of everything the model saw lies between 1 and 100 kPa. A grid that
is uniform in log ψ looks even on a plot, but the model does not see a plot —
it sees a set, and uniform log spacing puts a large share of that set into the
thinly-sampled wet and dry extremes.

## Grid comparison

| grid | <1 kPa | 1–10 | 10–100 | 100–1e3 | >1e3 |
|---|---|---|---|---|---|
| **training data** | 14.4 | 32.5 | 32.7 | 11.2 | 9.2 |
| **recommended (below)** | 15.8 | 31.6 | 31.6 | 10.5 | 10.5 |
| uniform log, 0.01–15000, 19 pts | 31.6 | 15.8 | 10.5 | 15.8 | 26.3 |
| former default, 1–15000, 12 pts | 0.0 | 25.0 | 16.7 | 25.0 | 33.3 |

The uniform-log grid over-weights below 1 kPa by more than a factor of two and
under-weights 10–100 kPa by a factor of three. The former default had nothing
below 1 kPa at all. Both are mismatched, in opposite directions.

## Recommended default

```
0.01, 0.1, 0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 33, 50, 70, 100, 300, 1500, 15000
```

19 points, 0.01 to 15000 kPa, density matched to training within ~1.5
percentage points in every band. Contains 0.01, 33 and 1500 kPa exactly, so
θ_sat, field capacity and permanent wilting point are read directly rather
than interpolated.

## Guidance

- Prefer this grid, or one with a similar density profile, over uniform log
  spacing.
- Keep the span wide. Narrowing it is a larger departure than re-spacing it.
- Avoid single-point requests; HABIT is a curve predictor.
- To reproduce a previously quoted number, reproduce the grid as well as the
  soil properties. This applies to the app, the `habit-ptf` package and batch
  runs alike.

## Limitations

The density figures above are pooled across all training samples. The model
conditions on each sample's own set, so matching the pooled marginal is a
coarse target — it removes the obvious mismatch but does not guarantee an
unbiased result. Validate against measured data before relying on it.

The single-soil tab builds its grid with `np.logspace(min, max, points)` and
therefore cannot express a non-uniform density. Its predictions carry the
uniform-log profile shown in the table above. The batch tab accepts an
arbitrary list and can use the recommended grid directly.
