# HABIT — Soil Water Retention Predictor

Interactive web app for **HABIT** (Hierarchical Attention-Based Inference with Transfer Learning), a pre-trained ensemble model for predicting soil water retention curves.

**Paper:** Ghezzehei TA (2026). *Water Resources Research*, 62, e2025WR042833.
[doi:10.1029/2025WR042833](https://doi.org/10.1029/2025WR042833)

**Model weights:** [Teamrat/habit](https://huggingface.co/Teamrat/habit)

## Live app

[**Launch HABIT Predictor**](https://soil-habit.streamlit.app) on Streamlit Community Cloud.

## Features

- Predict water retention curves from soil texture (required) plus optional bulk density, organic carbon, and saturated hydraulic conductivity
- 20-member ensemble with uncertainty quantification (mean ± 95% CI)
- Automatic input adaptation — the model detects which properties you provide
- Single soil prediction with interactive plot
- Batch prediction from CSV upload
- Download results as CSV

## Input units

Units are fixed and are **not** auto-detected or converted:

| Property | Units |
|----------|-------|
| Sand, silt, clay | percent by mass (summing to ~100) |
| Bulk density | g/cm³ |
| Organic carbon | percent by mass |
| Ksat | cm/day |

## How it works

The app uses ONNX Runtime for fast, lightweight inference (~50 MB vs ~700 MB for TensorFlow). Model weights are downloaded from HuggingFace on first launch and cached locally.

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Python package

For programmatic use:

```bash
pip install habit-ptf
```

```python
from habit_ptf import load_ensemble
predictor = load_ensemble()
result = predictor.predict(soil_dataframe)
```
