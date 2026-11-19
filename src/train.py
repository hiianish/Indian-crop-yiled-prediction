from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import load_dataset, add_history, feature_sets
from .models import estimators, make_pipeline


def score(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
            "mape_pct": float(np.mean(np.abs((np.asarray(y)[np.asarray(y) != 0] - np.asarray(pred)[np.asarray(y) != 0]) / np.asarray(y)[np.asarray(y) != 0])) * 100) if np.any(np.asarray(y) != 0) else None}


def main():
    parser = argparse.ArgumentParser(description="Compare crop yield regression models on chronological holdouts")
    parser.add_argument("--data", default=os.getenv("CROP_DATA", "data/processed/crop_yield.csv"))
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--test-fraction", type=float, default=.2)
    args = parser.parse_args()
    outdir = Path(args.artifacts); outdir.mkdir(parents=True, exist_ok=True)
    df = add_history(load_dataset(args.data))
    years = sorted(df.year.unique())
    if len(years) < 3:
        raise ValueError("At least three distinct years are required for train/validation/test chronology.")
    cutoff = max(1, int(len(years) * (1 - args.test_fraction)))
    test_years = years[cutoff:]
    train = df[df.year < min(test_years)].copy()
    test = df[df.year.isin(test_years)].copy()
    train = train[train.previous_yield.notna()]
    test = test[test.previous_yield.notna()]
    if train.empty or test.empty:
        raise ValueError("No eligible train/test rows after creating prior-yield features. Check data coverage.")
    results, predictions = [], []
    # Prior-year baseline: previous observed yield is already computed without target leakage.
    base_pred = test.previous_yield.to_numpy()
    results.append({"feature_set":"history", "model":"Previous yield baseline", **score(test.yield_t_ha, base_pred), "fit_seconds":0.0, "predict_seconds":0.0})
    predictions.append(pd.DataFrame({"row_id":test.index, "year":test.year, "crop":test.crop, "state":test.state, "district":test.district, "actual":test.yield_t_ha, "prediction":base_pred, "model":"Previous yield baseline", "feature_set":"history"}))
    for set_name, cols in feature_sets(df).items():
        cols = [c for c in cols if c in df.columns]
        for name, estimator in estimators().items():
            print(f"Fitting {name} on {set_name} ({len(train):,} training rows)...", flush=True)
            fit_data = train
            if name == "SVR" and len(train) > 2000:
                fit_data = train.sample(2000, random_state=42)
            pipe = make_pipeline(fit_data, cols, estimator)
            start = time.perf_counter(); pipe.fit(fit_data[cols], fit_data.yield_t_ha); fit_s = time.perf_counter()-start
            start = time.perf_counter(); pred = pipe.predict(test[cols]); predict_s = time.perf_counter()-start
            results.append({"feature_set":set_name, "model":name, "fit_rows":len(fit_data), **score(test.yield_t_ha, pred), "fit_seconds":fit_s, "predict_seconds":predict_s})
            predictions.append(pd.DataFrame({"row_id":test.index, "year":test.year, "crop":test.crop, "state":test.state, "district":test.district, "actual":test.yield_t_ha, "prediction":pred, "model":name, "feature_set":set_name}))
            pd.DataFrame(results).to_csv(outdir/"metrics.csv", index=False)
            if set_name == "history_weather_soil":
                joblib.dump({"pipeline":pipe, "features":cols, "model_name":name}, outdir / f"{name.lower().replace(' ','_')}.joblib")
    pd.DataFrame(results).to_csv(outdir/"metrics.csv", index=False)
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_csv(outdir/"predictions.csv", index=False)
    grouped = []
    for group_type in ["crop", "state"]:
        for (model, feature_set, group_name), frame in pred_df.groupby(["model", "feature_set", group_type]):
            grouped.append({"model": model, "feature_set": feature_set, "group_type": group_type,
                            "group": group_name, **score(frame.actual, frame.prediction)})
    pd.DataFrame(grouped).to_csv(outdir/"metrics_by_crop_state.csv", index=False)
    best = pd.DataFrame(results).sort_values("rmse").iloc[0]
    report = {"train_years": [int(x) for x in sorted(train.year.unique())], "test_years": [int(x) for x in test_years], "train_rows":len(train), "test_rows":len(test), "best_by_rmse": {k: (float(v) if isinstance(v, (np.floating, float)) else str(v)) for k,v in best.to_dict().items()}}
    (outdir/"run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
