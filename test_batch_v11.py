"""Tests for batch_v11 — docs/app-design.md section 12.2.

Runs on numpy and pandas alone: no Streamlit, no onnxruntime, no weights.

    python test_batch_v11.py
"""

import io
import sys

import numpy as np
import pandas as pd

import batch_v11 as B

FAILURES = []


def check(name, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def expect_error(name, fn, *must_mention):
    try:
        fn()
    except B.BatchError as e:
        msg = str(e)
        missing = [m for m in must_mention if m not in msg]
        check(name, not missing,
              f"names {must_mention}" if not missing
              else f"message omits {missing}: {msg!r}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"raised {type(e).__name__} instead of BatchError")
    else:
        check(name, False, "no error raised")


def csv(text):
    return pd.read_csv(io.StringIO(text))


# ── A stub model: theta linear in log10 psi, so interpolation is exact ─────

def stub_curves(soils, n_members=20, seed=0):
    """(n_members, n_soils, 151). Linear in log10 psi by construction."""
    rng = np.random.RandomState(seed)
    clay = soils["clay"].to_numpy(dtype=float)
    a = 0.45 + clay / 500.0
    b = 0.03 + clay / 5000.0
    member_offset = rng.uniform(-0.01, 0.01, n_members)
    out = np.empty((n_members, len(soils), B.GRID_LOG10.size))
    for m in range(n_members):
        for s in range(len(soils)):
            out[m, s] = a[s] + member_offset[m] - b[s] * B.GRID_LOG10
    return out


WIDE = B.TEMPLATE_WIDE
LONG = B.TEMPLATE_LONG


# ── 1. Interpolation ──────────────────────────────────────────────────────

def test_interpolation():
    print("\ninterpolation")
    soils = B.parse_batch(csv(WIDE)).soils
    curves = stub_curves(soils, n_members=1)

    at_nodes = B.interpolate_members(curves, B.GRID_KPA)
    check("exact at every grid node",
          np.allclose(at_nodes[0], curves[0], atol=0, rtol=0),
          f"max diff {np.max(np.abs(at_nodes[0] - curves[0])):.1e}")

    psi = np.array([0.03, 0.7, 33.0, 1234.0, 15000.0])
    got = B.interpolate_members(curves, psi)[0, 0]
    slope = (curves[0, 0, -1] - curves[0, 0, 0]) / (B.GRID_LOG10[-1]
                                                    - B.GRID_LOG10[0])
    want = curves[0, 0, 0] + slope * (np.log10(psi) - B.GRID_LOG10[0])
    check("exact on a function linear in log10 psi",
          np.allclose(got, want, atol=1e-12),
          f"max diff {np.max(np.abs(got - want)):.1e}")

    # To float64 precision, not bitwise: the grid is uniform, so the bracket is
    # arithmetic rather than a search, and it reaches the same value by a
    # different route. Measured worst case across the domain, on nodes and at
    # both endpoints: 2.3e-16, which is machine epsilon.
    ref = np.interp(np.log10(psi), B.GRID_LOG10, curves[0, 0])
    check("agrees with numpy.interp", np.allclose(got, ref, atol=1e-14, rtol=0),
          f"max diff {np.max(np.abs(got - ref)):.1e}")

    check("endpoints are inside the domain",
          B.PSI_MIN_KPA == 1e-2 and abs(B.PSI_MAX_KPA - 10 ** 5.5) < 1,
          f"{B.PSI_MIN_KPA:g} to {B.PSI_MAX_KPA:.6g} kPa")
    check("GRID_ID matches the grid",
          B.GRID_ID == "logpsi_-2.0_5.5_151" and B.GRID_LOG10.size == 151)


# ── 2. Aggregation order ──────────────────────────────────────────────────

def test_aggregation_order():
    print("\naggregation order")
    soils = B.parse_batch(csv(WIDE)).soils
    curves = stub_curves(soils)
    psi = np.array([0.01, 33.0, 1500.0])

    interp_first = B.aggregate(B.interpolate_members(curves, psi))["theta"]
    mean_first = B.interpolate_members(curves.mean(axis=0)[None, ...],
                                       psi)[0]
    check("mean is invariant to the order",
          np.allclose(interp_first, mean_first, atol=1e-12),
          f"max diff {np.max(np.abs(interp_first - mean_first)):.1e}")

    sd = B.aggregate(B.interpolate_members(curves, psi))["theta_sd"]
    check("sd is finite and non-negative",
          np.all(np.isfinite(sd)) and np.all(sd >= 0))


# ── 3. Grid independence, end to end ──────────────────────────────────────

def test_grid_independence():
    print("\ngrid independence")
    parsed = B.parse_batch(csv(WIDE))
    curves = stub_curves(parsed.soils)

    sets = [[33.0],
            [0.01, 33.0, 1500.0],
            [0.01, 0.1, 1, 10, 33, 100, 1500, 15000]]
    thetas = []
    for s in sets:
        out = B.build_output(parsed, curves, psi_kpa=s)
        row = out[(out["soil_id"] == "sample_1") & (out["psi_kpa"] == 33.0)]
        thetas.append(float(row["theta"].iloc[0]))

    check("theta at 33 kPa is identical for 1, 3 and 8 requested points",
          thetas[0] == thetas[1] == thetas[2],
          f"{thetas[0]!r}, {thetas[1]!r}, {thetas[2]!r}")


# ── 4. Schema errors — one per row of section 12.1 ────────────────────────

def test_schema_errors():
    print("\nschema errors")
    good = WIDE

    expect_error("unknown column named, with the accepted set",
                 lambda: B.parse_batch(csv(good.replace("clay", "clay_pct"))),
                 "clay_pct", "Accepted columns")

    expect_error("missing predictor header named",
                 lambda: B.parse_batch(csv(
                     "soil_id,sand,silt,clay,bd,oc\ns1,40,40,20,1.3,1.0\n")),
                 "ksat")

    expect_error("missing soil_id",
                 lambda: B.parse_batch(csv(
                     "sand,silt,clay,bd,oc,ksat\n40,40,20,1.3,1.0,10\n")),
                 "soil_id")

    expect_error("blank texture names the row",
                 lambda: B.parse_batch(csv(
                     "soil_id,sand,silt,clay,bd,oc,ksat\ns1,,40,20,1.3,1.0,10\n")),
                 "Texture is blank", "2")

    expect_error("texture sum wrong names row and sum",
                 lambda: B.parse_batch(csv(
                     "soil_id,sand,silt,clay,bd,oc,ksat\ns1,0.4,0.4,0.2,1.3,1,10\n")),
                 "sum to ~100", "1")

    expect_error("theta_obs without psi_kpa",
                 lambda: B.parse_batch(csv(
                     "soil_id,sand,silt,clay,bd,oc,ksat,theta_obs\n"
                     "s1,40,40,20,1.3,1.0,10,0.3\n")),
                 "theta_obs", "psi_kpa")

    expect_error("duplicate soil_id in wide form names the id",
                 lambda: B.parse_batch(csv(
                     "soil_id,sand,silt,clay,bd,oc,ksat\n"
                     "s1,40,40,20,1.3,1.0,10\ns1,50,30,20,1.3,1.0,10\n")),
                 "s1", "Repeated soil_id")

    expect_error("conflicting predictors name the id and the column",
                 lambda: B.parse_batch(csv(
                     "soil_id,sand,silt,clay,bd,oc,ksat,psi_kpa\n"
                     "s1,40,40,20,1.30,1.0,10,33\n"
                     "s1,40,40,20,1.45,1.0,10,1500\n")),
                 "s1", "bd")

    expect_error("psi out of domain names the value and the range",
                 lambda: B.parse_batch(csv(
                     "soil_id,sand,silt,clay,bd,oc,ksat,psi_kpa\n"
                     "s1,40,40,20,1.3,1.0,10,1e6\n")),
                 "outside the model's range")

    expect_error("unparseable psi list names the token",
                 lambda: B.parse_psi_list("10, thirty-three, 1500"),
                 "thirty-three")

    check("a clean wide file parses", B.parse_batch(csv(good)).form == "wide")
    check("a clean long file parses", B.parse_batch(csv(LONG)).form == "long")


# ── 5. Long-form round trip ───────────────────────────────────────────────

def test_long_round_trip():
    print("\nlong-form round trip")
    src = csv(LONG)
    parsed = B.parse_batch(src)
    curves = stub_curves(parsed.soils)
    out = B.build_output(parsed, curves)

    check("one output row per input row", len(out) == len(src))
    check("input row order preserved",
          list(out["soil_id"]) == [str(s) for s in src["soil_id"]]
          and np.allclose(out["psi_kpa"], src["psi_kpa"]))
    check("theta_obs passed through unmodified",
          np.allclose(out["theta_obs"].to_numpy(dtype=float),
                      src["theta_obs"].to_numpy(dtype=float)))
    check("note column present", "note" in out.columns)
    check("predictors echoed", all(c in out.columns for c in B.PREDICTORS))
    check("stage is an integer 0..3",
          out["stage"].isin([0, 1, 2, 3]).all(),
          f"values {sorted(out['stage'].unique())}")
    check("sample_2 is stage 1 (bd only, no oc)",
          int(out[out["soil_id"] == "sample_2"]["stage"].iloc[0]) == 1)
    check("sample_1 is stage 3 (everything)",
          int(out[out["soil_id"] == "sample_1"]["stage"].iloc[0]) == 3)


# ── 6. Deduplication is order-independent ─────────────────────────────────

def test_dedup_order():
    print("\ndeduplication")
    src = csv(LONG)
    shuffled = src.iloc[[3, 0, 4, 2, 1]].reset_index(drop=True)

    a = B.build_output(B.parse_batch(src), stub_curves(
        B.parse_batch(src).soils))
    p_shuf = B.parse_batch(shuffled)
    b = B.build_output(p_shuf, stub_curves(p_shuf.soils))

    merged = a.merge(b, on=["soil_id", "psi_kpa"], suffixes=("_a", "_b"))
    check("every (soil, psi) pair appears in both", len(merged) == len(src))
    check("shuffling the rows does not change any answer",
          np.allclose(merged["theta_a"], merged["theta_b"], atol=0, rtol=0),
          f"max diff {np.max(np.abs(merged['theta_a'] - merged['theta_b'])):.1e}")


# ── 7. Wide form output shape ─────────────────────────────────────────────

def test_wide_output():
    print("\nwide form output")
    parsed = B.parse_batch(csv(WIDE))
    curves = stub_curves(parsed.soils)
    psi = [1500.0, 33.0, 0.01]                    # deliberately unsorted
    out = B.build_output(parsed, curves, psi_kpa=psi)

    check("rows = soils x potentials", len(out) == 3 * 3)
    check("output is long form even for one potential",
          len(B.build_output(parsed, curves, psi_kpa=[33.0])) == 3)
    check("soils in input order",
          list(out["soil_id"].unique()) == ["sample_1", "sample_2", "sample_3"])
    check("potentials ascending within a soil",
          list(out[out["soil_id"] == "sample_1"]["psi_kpa"]) == [0.01, 33.0, 1500.0])
    check("note repeats across a soil's rows",
          list(out[out["soil_id"] == "sample_3"]["note"]) == ["coarse"] * 3)
    check("no theta_obs column when not supplied",
          "theta_obs" not in out.columns)
    check("sample_3 is stage 0 (texture only)",
          int(out[out["soil_id"] == "sample_3"]["stage"].iloc[0]) == 0)

    attn = np.tile(np.array([[0.4, 0.3, 0.2, 0.1]]), (3, 1))
    with_attn = B.build_output(parsed, curves, psi_kpa=psi, attention=attn)
    check("attention columns appended when given",
          all(c in with_attn.columns for c in
              ["attn_texture", "attn_bd", "attn_oc", "attn_ksat"]))


def main():
    for fn in (test_interpolation, test_aggregation_order,
               test_grid_independence, test_schema_errors,
               test_long_round_trip, test_dedup_order, test_wide_output):
        fn()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("All checks pass.")


if __name__ == "__main__":
    main()
