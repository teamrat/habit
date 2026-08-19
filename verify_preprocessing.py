"""Verify app.py preprocessing reproduces the archived HABIT training arrays.

Extracts the preprocessing functions from app.py (without importing Streamlit)
and checks their output against HABIT-WRR-dryad/data/processed/, which pairs
raw soil properties (*_original_data.json) with the exact scaled tensors the
model was trained on (*_data.npz).
"""
import ast
import json
import sys

import numpy as np
import pandas as pd

APP = "/tmp/work/app.py"
ARCHIVE = "/mnt/user-data/uploads/habit_dev/HABIT-WRR-dryad/data/processed"

WANTED = {
    "SCALER_PARAMS",
    "robust_scale", "transform_oc", "transform_ksat",
    "prepare_inputs", "prepare_batch_inputs",
}

# Pull the target definitions straight out of app.py so we test shipped code.
tree = ast.parse(open(APP).read())
keep = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef,)) and node.name in WANTED:
        keep.append(node)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in WANTED:
                keep.append(node)

ns = {"np": np, "pd": pd}
exec(compile(ast.Module(body=keep, type_ignores=[]), APP, "exec"), ns)

missing = WANTED - set(ns)
if missing:
    sys.exit(f"FAIL: could not extract {missing} from app.py")

# The app emits float32 tensors (ONNX input dtype); the archive is float64,
# so agreement is bounded by float32 resolution, not by the transform.
TOL = 1e-6
failures = []


def check(name, got, want):
    err = float(np.nanmax(np.abs(np.asarray(got) - np.asarray(want))))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  [{status}] {name:<34} max abs err = {err:.3e}")
    if status == "FAIL":
        failures.append(name)


for level in ["level0", "level1", "level2", "level3"]:
    npz = np.load(f"{ARCHIVE}/{level}/train_data.npz", allow_pickle=True)
    orig = json.load(open(f"{ARCHIVE}/{level}/train_original_data.json"))
    assert [o["sample_id"] for o in orig] == list(npz["soil_ids"])

    has_bd = any("bd" in o for o in orig)
    has_oc = any("oc_pct" in o for o in orig)
    has_ksat = any("ksat" in o for o in orig)
    print(f"\n{level}  (n={len(orig)}, bd={has_bd} oc={has_oc} ksat={has_ksat})")

    # ---- single-soil path: prepare_inputs, one soil at a time -------------
    wp = np.array([1.0, 33.0, 1500.0])
    tex_got, bd_got, oc_got, ksat_got = [], [], [], []
    for o in orig:
        # archive stores texture as fractions; the app takes percent
        feed, mask = prepare_inputs_args = ns["prepare_inputs"](
            o["sand"] * 100, o["silt"] * 100, o["clay"] * 100,
            o.get("bd"), o.get("oc_pct"), o.get("ksat"),
            wp,
        )
        tex_got.append(feed["texture"][0])
        bd_got.append(feed["bd"][0])
        oc_got.append(feed["oc"][0])
        ksat_got.append(feed["ksat"][0])

    check("prepare_inputs  texture", tex_got, npz["texture"])
    if has_bd:
        check("prepare_inputs  bd", bd_got, npz["bd"])
    if has_oc:
        check("prepare_inputs  oc", oc_got, npz["oc"])
    if has_ksat:
        check("prepare_inputs  ksat", ksat_got, npz["ksat"])

    # ---- batch path: prepare_batch_inputs on the whole level --------------
    rows = {
        "sand": [o["sand"] * 100 for o in orig],
        "silt": [o["silt"] * 100 for o in orig],
        "clay": [o["clay"] * 100 for o in orig],
    }
    if has_bd:
        rows["bd"] = [o.get("bd", np.nan) for o in orig]
    if has_oc:
        rows["oc"] = [o.get("oc_pct", np.nan) for o in orig]
    if has_ksat:
        rows["ksat"] = [o.get("ksat", np.nan) for o in orig]
    df = pd.DataFrame(rows)
    cols_lower = {c: c for c in df.columns}
    feed, mask = ns["prepare_batch_inputs"](df, cols_lower, wp)

    check("prepare_batch_inputs  texture", feed["texture"], npz["texture"])
    if has_bd:
        check("prepare_batch_inputs  bd", feed["bd"], npz["bd"])
    if has_oc:
        check("prepare_batch_inputs  oc", feed["oc"], npz["oc"])
    if has_ksat:
        check("prepare_batch_inputs  ksat", feed["ksat"], npz["ksat"])

print("\n" + "=" * 62)
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All preprocessing checks reproduce the training arrays exactly.")
