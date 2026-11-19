"""Permutation importance for the four selected models on the temporal test set."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from .data import add_history, load_dataset

MODELS = ["mlp", "ridge", "linear_regression", "catboost"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/crop_yield.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    data = add_history(load_dataset(args.data))
    years = sorted(data.year.unique())
    cutoff = max(1, int(len(years) * .8))
    test_years = years[cutoff:]
    test = data[data.year.isin(test_years) & data.previous_yield.notna()].copy()

    rows = []
    for name in MODELS:
        model_path = Path(args.artifacts) / f"{name}.joblib"
        if not model_path.exists():
            print(f"Skipping {name}: no saved model at {model_path}")
            continue
        bundle = joblib.load(model_path)
        columns = bundle["features"]
        result = permutation_importance(
            bundle["pipeline"],
            test[columns],
            test["yield_t_ha"],
            scoring="neg_root_mean_squared_error",
            n_repeats=args.repeats,
            random_state=42,
            n_jobs=1,
        )
        for feature, mean, std in zip(columns, result.importances_mean, result.importances_std):
            rows.append({
                "model": bundle["model_name"],
                "feature": feature,
                "rmse_increase_when_shuffled": float(mean),
                "std_across_shuffles": float(std),
                "interpretation": "helpful in this test" if mean > 0 else "no measured benefit in this test",
            })

    output = Path(args.artifacts) / "feature_importance.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows).sort_values(
        ["model", "rmse_increase_when_shuffled"], ascending=[True, False]
    )
    report.to_csv(output, index=False)
    print(f"Saved {len(report)} feature/model rows to {output}")
    for model, group in report.groupby("model", sort=False):
        print(f"\n{model}")
        print(group.head(8)[["feature", "rmse_increase_when_shuffled", "interpretation"]].to_string(index=False))


if __name__ == "__main__":
    main()
