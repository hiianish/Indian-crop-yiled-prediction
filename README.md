# Indian Crop Yield Prediction

Reproducible starter project for comparing crop-yield models across Indian crops and regions. The pipeline is designed around **district × crop × season × agricultural year** observations. It does not bundle official data: obtain and license source data from DES/Agmarknet, IMD and SoilGrids, then prepare it in the schema below.

## Quick start

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

I fetched a public, multi-source India district panel into `data/raw/`. It has 198,369 rows (1966–2017), 20 states and 21 crops; preparation currently selects Rice, Wheat, Maize, Cotton and Sugarcane. Run:

```powershell
python -m src.prepare_data
```

The source reports yield directly in kg/ha, so preparation converts that target to tonnes/ha. This starter source does not contain area and production columns; the pipeline also accepts the project's original schema if you later replace it with an official production/area panel. To use another prepared file, set `CROP_DATA` or pass `--data`.

Minimum canonical columns for a production/area source:

| Column | Meaning |
|---|---|
| `state`, `district`, `crop`, `season`, `year` | Observation keys; year is the starting year of the agricultural year |
| `production_tonnes`, `area_hectares` | Production and harvested/cultivated area, in the stated units |
| `rainfall_mm` | Optional season/year rainfall aggregate |
| `soil_*` | Optional numeric soil attributes (e.g. `soil_ph`, `soil_clay`) |

For production/area sources, yield is derived as production / area. The fetched source already supplies yield. One row per key is required. See `data/README.md` for provenance, caveats and leakage guidance.

```powershell
python -m src.train --data data/processed/crop_yield.csv
streamlit run app.py
```

The training command writes metrics, predictions and fitted pipelines under `artifacts/`. Optional XGBoost and CatBoost models are included when installed. Metrics include MAE, RMSE, R² and MAPE (only for nonzero targets); results are reported overall and by crop/state. The chronological test years are the latest 20% of distinct years. The report also includes history-only, +weather and +soil/input feature ablations when those columns exist. Kernel SVR is fit on a deterministic 2,000-row training sample to keep runtime manageable; `fit_rows` is recorded so that this limitation is visible in the comparison.

See [`artifacts/initial_results.md`](artifacts/initial_results.md) for the first completed comparison and caveats.

## What this starter implements

- Shared preprocessing and model comparison: Linear/Ridge, Decision Tree, Random Forest, XGBoost, CatBoost, SVR and MLP, plus a previous-yield baseline.
- Leakage-aware prior-yield feature from the previous available year within state/district/crop/season. Never uses the current row or future rows.
- Chronological holdout; overall metrics, per-crop/state metrics, timing and per-row test predictions.
- A Streamlit interface for dataset exploration and predictions from saved models, with feature importance for tree models.

## Important research choices

Treat this as a project scaffold, not a claim that any model is best. The starter panel has material limitations: temperature/humidity for 1966–1980 are climatological normals, pesticide is a national series broadcast across districts, pH is inferred from soil taxonomy, and weather is annual rather than crop-season-specific. These may make the features poor stand-ins for farm conditions. The temporal test measures future-year performance in represented regions, and the by-crop/state report breaks down those errors. A separate unseen-district holdout and strict future-plus-unseen-region test should be added before making spatial generalization claims. Tune on training years only; preserve the final test years until model selection is complete. SHAP, Optuna, raster processing and MLflow are planned extensions, not silently simulated results.

## Layout

`src/data.py` validation and feature engineering · `src/models.py` model definitions · `src/train.py` experiments · `app.py` interactive UI.
