"""Batch CSV handling for HABIT v1.1.

Everything here is model-free: parsing, validation, deduplication,
interpolation and output assembly. The caller runs the ensemble between
`parse_batch` and `build_output`, which keeps this module testable without
onnxruntime, Streamlit or any weights.

Implements docs/app-design.md sections 3.3, 4.2, 4.4, 6 and 12.1. Where this
file and that document disagree, the document is right and this is a bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── The canonical grid (docs/app-design.md 1.2) ───────────────────────────
# Part of the model definition. The endpoints are the clip bounds enforced in
# training, so no caller request can fall outside them.

GRID_ID = "logpsi_-2.0_5.5_151"
GRID_LOG10 = np.linspace(-2.0, 5.5, 151)
GRID_KPA = 10.0 ** GRID_LOG10
PSI_MIN_KPA = float(GRID_KPA[0])       # 1.0e-2
PSI_MAX_KPA = float(GRID_KPA[-1])      # 3.162278e5

# ── Schema (6.1) ──────────────────────────────────────────────────────────

TEXTURE = ["sand", "silt", "clay"]
OPTIONAL_PREDICTORS = ["bd", "oc", "ksat"]
PREDICTORS = TEXTURE + OPTIONAL_PREDICTORS
PASSTHROUGH = ["theta_obs", "note"]
ALLOWED = set(["soil_id"] + PREDICTORS + ["psi_kpa"] + PASSTHROUGH)

TEXTURE_SUM_TOLERANCE = 2.0            # percent, either side of 100

DEFAULT_PSI = [0.01, 0.1, 0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 33, 50, 70,
               100, 300, 1500, 15000]


class BatchError(ValueError):
    """A problem with the user's file. The message names the cause."""


def _rows(index) -> str:
    """Human row numbers, as they appear in a spreadsheet (header is row 1)."""
    nums = [int(i) + 2 for i in np.atleast_1d(index)]
    shown = ", ".join(str(n) for n in nums[:8])
    return shown + (f" (+{len(nums) - 8} more)" if len(nums) > 8 else "")


# ── Parsing and validation (6.1, 6.2, 12.1) ───────────────────────────────

class ParsedBatch:
    """A validated batch, ready for prediction.

    Attributes
    ----------
    form : "wide" | "long"
    frame : pd.DataFrame
        The input, with normalised column names and numeric columns coerced.
    soils : pd.DataFrame
        One row per unique soil, in first-appearance order: soil_id plus the
        six predictors. This is what gets predicted.
    soil_index : np.ndarray
        For each input row, the position of its soil in `soils`.
    """

    def __init__(self, form, frame, soils, soil_index):
        self.form = form
        self.frame = frame
        self.soils = soils
        self.soil_index = soil_index

    @property
    def n_soils(self) -> int:
        return len(self.soils)


def parse_batch(df: pd.DataFrame) -> ParsedBatch:
    """Validate a batch CSV and reduce it to unique soils.

    Raises BatchError, naming the cause, for every condition in
    docs/app-design.md 12.1. Nothing is silently corrected.
    """
    frame = df.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]

    unknown = [c for c in frame.columns if c not in ALLOWED]
    if unknown:
        raise BatchError(
            f"Unrecognised column(s): {', '.join(unknown)}. "
            f"Accepted columns are: {', '.join(sorted(ALLOWED))}. "
            f"Column names are fixed; nothing outside this set is accepted.")

    if len(set(frame.columns)) != len(frame.columns):
        dupes = sorted({c for c in frame.columns
                        if list(frame.columns).count(c) > 1})
        raise BatchError(f"Duplicate column(s): {', '.join(dupes)}.")

    if "soil_id" not in frame.columns:
        raise BatchError("Missing required column: soil_id.")

    missing_headers = [c for c in PREDICTORS if c not in frame.columns]
    if missing_headers:
        raise BatchError(
            f"Missing predictor column header(s): {', '.join(missing_headers)}. "
            f"The header must carry every predictor column; leave cells blank "
            f"where a property is unavailable.")

    if "theta_obs" in frame.columns and "psi_kpa" not in frame.columns:
        raise BatchError(
            "theta_obs is present but psi_kpa is not. An observed water "
            "content cannot be placed on the curve without the potential it "
            "was measured at.")

    if frame.empty:
        raise BatchError("The file has a header but no rows.")

    for col in PREDICTORS + (["psi_kpa"] if "psi_kpa" in frame.columns else [])\
            + (["theta_obs"] if "theta_obs" in frame.columns else []):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if frame["soil_id"].isna().any():
        raise BatchError(f"soil_id is blank on row(s) "
                         f"{_rows(frame.index[frame['soil_id'].isna()])}.")
    frame["soil_id"] = frame["soil_id"].astype(str)

    # Texture must carry values (6.1).
    blank_texture = frame[TEXTURE].isna().any(axis=1)
    if blank_texture.any():
        raise BatchError(
            f"Texture is blank on row(s) {_rows(frame.index[blank_texture])}. "
            f"Sand, silt and clay are required; there is no prediction "
            f"without them.")

    totals = frame[TEXTURE].sum(axis=1)
    bad = (totals - 100.0).abs() > TEXTURE_SUM_TOLERANCE
    if bad.any():
        first = frame.index[bad][0]
        raise BatchError(
            f"Texture does not sum to ~100 on row(s) {_rows(frame.index[bad])} "
            f"— row {int(first) + 2} sums to {totals[first]:.3g}. "
            f"Sand, silt and clay are percentages by mass; fractions are not "
            f"accepted.")

    form = "long" if "psi_kpa" in frame.columns else "wide"

    if form == "long":
        if frame["psi_kpa"].isna().any():
            raise BatchError(f"psi_kpa is blank on row(s) "
                             f"{_rows(frame.index[frame['psi_kpa'].isna()])}.")
        check_psi_domain(frame["psi_kpa"].to_numpy())

    ids = frame["soil_id"].to_numpy()
    first_pos = {}
    for i, sid in enumerate(ids):
        first_pos.setdefault(sid, i)
    order = sorted(first_pos, key=first_pos.get)

    if form == "wide" and len(order) != len(frame):
        counts = pd.Series(ids).value_counts()
        repeated = sorted(counts[counts > 1].index)
        raise BatchError(
            f"Repeated soil_id in a wide file: {', '.join(map(str, repeated))}. "
            f"Without a psi_kpa column each soil must appear exactly once.")

    # Long form: every row of a soil must carry identical predictors (6.2).
    if form == "long":
        for sid, group in frame.groupby("soil_id", sort=False):
            if len(group) == 1:
                continue
            for col in PREDICTORS:
                vals = group[col]
                uniq = vals.dropna().unique()
                inconsistent = len(uniq) > 1 or (
                    vals.isna().any() and len(uniq) == 1)
                if inconsistent:
                    raise BatchError(
                        f"Soil '{sid}' has different values of '{col}' on "
                        f"different rows (rows {_rows(group.index)}). Every "
                        f"row of a soil must carry identical predictors.")

    pos = {sid: k for k, sid in enumerate(order)}
    soil_index = np.array([pos[s] for s in ids], dtype=int)

    soils = (frame.loc[[frame.index[first_pos[s]] for s in order],
                       ["soil_id"] + PREDICTORS]
             .reset_index(drop=True))

    return ParsedBatch(form, frame, soils, soil_index)


def check_psi_domain(psi_kpa) -> np.ndarray:
    """Reject potentials outside the canonical grid (4.4). Never clamps."""
    psi = np.asarray(psi_kpa, dtype=float)
    if psi.size == 0:
        raise BatchError("No water potentials given.")
    if not np.all(np.isfinite(psi)):
        raise BatchError("Water potentials must be finite numbers.")
    if np.any(psi <= 0):
        raise BatchError("Water potentials are magnitudes in kPa and must be "
                         "positive.")
    outside = (psi < PSI_MIN_KPA) | (psi > PSI_MAX_KPA)
    if np.any(outside):
        raise BatchError(
            f"Water potential {psi[outside][0]:g} kPa is outside the model's "
            f"range of {PSI_MIN_KPA:g} to {PSI_MAX_KPA:.6g} kPa. Values "
            f"outside it are refused rather than clamped.")
    return psi


def parse_psi_list(text: str) -> np.ndarray:
    """Parse the UI's comma-separated potentials, ascending and de-duplicated."""
    tokens = [t.strip() for t in str(text).split(",") if t.strip()]
    if not tokens:
        raise BatchError("Enter at least one water potential.")
    values = []
    for tok in tokens:
        try:
            values.append(float(tok))
        except ValueError:
            raise BatchError(f"Could not read '{tok}' as a water potential. "
                             f"Use comma-separated numbers in kPa.")
    return check_psi_domain(np.array(sorted(set(values))))


# ── Stage (3.3) ───────────────────────────────────────────────────────────

def stage_of(row) -> int:
    """Highest nested level fully satisfied: 0 texture, 1 +BD, 2 +OC, 3 +Ksat.

    A soil with organic carbon but no bulk density is stage 0 by this rule.
    Nothing is lost — the output echoes every predictor column, so the blank
    cells state exactly what was supplied.
    """
    level = 0
    for name in OPTIONAL_PREDICTORS:
        if pd.isna(row.get(name)):
            break
        level += 1
    return level


def stages_for(soils: pd.DataFrame) -> np.ndarray:
    return np.array([stage_of(r) for _, r in soils.iterrows()], dtype=int)


# ── Interpolation and aggregation (4.2) ───────────────────────────────────

def _bracket(log_psi):
    """Grid cell index and blend weight for each value, arithmetically.

    The grid is uniform in log10 psi, so the bracketing index is a division —
    no searchsorted, no per-value Python. Values are clipped to the domain;
    check_psi_domain is what rejects anything outside it, before this runs.
    """
    span = GRID_LOG10[-1] - GRID_LOG10[0]
    spacing = span / (GRID_LOG10.size - 1)
    pos = np.clip((np.asarray(log_psi, dtype=float) - GRID_LOG10[0]) / spacing,
                  0.0, GRID_LOG10.size - 1)
    i0 = np.clip(np.floor(pos), 0, GRID_LOG10.size - 2).astype(np.intp)
    return i0, pos - i0


def interpolate_members(curves: np.ndarray, psi_kpa) -> np.ndarray:
    """Interpolate each member's curve, linearly in log10 psi.

    Parameters
    ----------
    curves : (n_members, n_soils, 151)
        Model output on the canonical grid.
    psi_kpa : (n_points,) or (n_soils, n_points)
        Output potentials. A 1-D array is shared by every soil; a 2-D array
        gives each soil its own.

    Returns
    -------
    (n_members, n_soils, n_points)
    """
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 3 or curves.shape[-1] != GRID_LOG10.size:
        raise ValueError(f"curves must be (n_members, n_soils, "
                         f"{GRID_LOG10.size}); got {curves.shape}")
    n_members, n_soils, _ = curves.shape

    psi = np.asarray(psi_kpa, dtype=float)
    if psi.ndim == 1:
        psi = np.tile(psi, (n_soils, 1))
    if psi.shape[0] != n_soils:
        raise ValueError(f"psi has {psi.shape[0]} rows, expected {n_soils}")
    check_psi_domain(psi.ravel())

    i0, w = _bracket(np.log10(psi))            # both (n_soils, n_points)
    rows = np.arange(n_soils)[:, None]
    y0 = curves[:, rows, i0]                   # (n_members, n_soils, n_points)
    y1 = curves[:, rows, i0 + 1]
    return y0 + (y1 - y0) * w


def aggregate(member_values: np.ndarray) -> dict:
    """Mean, sd and the 2.5/97.5% quantiles across members.

    Interpolation happens first, aggregation second. The mean is identical
    either way — linear interpolation commutes with averaging — but the sd is
    not, and this order matches what a per-point query would give.
    """
    v = np.asarray(member_values, dtype=float)
    return {
        "theta": v.mean(axis=0),
        "theta_sd": v.std(axis=0, ddof=0),
        "theta_q025": np.quantile(v, 0.025, axis=0),
        "theta_q975": np.quantile(v, 0.975, axis=0),
    }


# ── Output assembly (6.3) ─────────────────────────────────────────────────

def build_output(parsed: ParsedBatch,
                 curves: np.ndarray,
                 psi_kpa=None,
                 attention: np.ndarray | None = None) -> pd.DataFrame:
    """Assemble the long-form result.

    Parameters
    ----------
    parsed : ParsedBatch
    curves : (n_members, n_soils, 151)
        Canonical-grid predictions for `parsed.soils`, in that order.
    psi_kpa : array-like, wide form only
        The common output potentials. Ignored for long form, which uses each
        row's own psi_kpa.
    attention : (n_soils, 4), optional
        Property attention weights, appended as four columns when given.
    """
    frame, soils = parsed.frame, parsed.soils
    stages = stages_for(soils)

    if parsed.form == "long":
        psi_per_row = frame["psi_kpa"].to_numpy(dtype=float)
        # One column per row, gathered back per soil, so each member is
        # interpolated at exactly the potential its row asked for.
        check_psi_domain(psi_per_row)
        i0, w = _bracket(np.log10(psi_per_row))          # (n_rows,)
        soil = np.asarray(parsed.soil_index)
        y0 = curves[:, soil, i0]                         # (n_members, n_rows)
        y1 = curves[:, soil, i0 + 1]
        agg = aggregate(y0 + (y1 - y0) * w)

        out = pd.DataFrame({"soil_id": frame["soil_id"].to_numpy()})
        for col in PREDICTORS:
            out[col] = frame[col].to_numpy()
        out["psi_kpa"] = psi_per_row
        for k, v in agg.items():
            out[k] = v
        out["stage"] = stages[parsed.soil_index]
        row_soil = parsed.soil_index

    else:
        psi = check_psi_domain(np.asarray(psi_kpa, dtype=float))
        psi = np.sort(np.unique(psi))
        vals = interpolate_members(curves, psi)               # (M, S, P)
        agg = {k: v.ravel() for k, v in
               aggregate(vals).items()}                       # soil-major
        n_psi = psi.size

        out = pd.DataFrame({
            "soil_id": np.repeat(soils["soil_id"].to_numpy(), n_psi)})
        for col in PREDICTORS:
            out[col] = np.repeat(soils[col].to_numpy(), n_psi)
        out["psi_kpa"] = np.tile(psi, parsed.n_soils)
        for k, v in agg.items():
            out[k] = v
        out["stage"] = np.repeat(stages, n_psi)
        row_soil = np.repeat(np.arange(parsed.n_soils), n_psi)

    for col in PASSTHROUGH:
        if col in frame.columns:
            values = frame[col].to_numpy()
            out[col] = values if parsed.form == "long" else values[
                [np.where(parsed.soil_index == s)[0][0] for s in row_soil]]

    if attention is not None:
        attention = np.asarray(attention, dtype=float)
        for j, name in enumerate(["attn_texture", "attn_bd",
                                  "attn_oc", "attn_ksat"]):
            out[name] = attention[row_soil, j]

    return out


# ── Templates (6.5) ───────────────────────────────────────────────────────
# theta_obs values below are illustrative placeholders, not measurements.

TEMPLATE_WIDE = (
    "soil_id,sand,silt,clay,bd,oc,ksat,note\n"
    "sample_1,40,40,20,1.35,1.2,50,\n"
    "sample_2,65,25,10,1.50,,,\n"
    "sample_3,90,5,5,,,,coarse\n"
)

TEMPLATE_LONG = (
    "soil_id,sand,silt,clay,bd,oc,ksat,psi_kpa,theta_obs,note\n"
    "sample_1,40,40,20,1.35,1.2,50,10,0.312,\n"
    "sample_1,40,40,20,1.35,1.2,50,33,0.268,\n"
    "sample_1,40,40,20,1.35,1.2,50,1500,0.141,\n"
    "sample_2,65,25,10,1.50,,,33,0.191,\n"
    "sample_2,65,25,10,1.50,,,1500,0.087,\n"
)
