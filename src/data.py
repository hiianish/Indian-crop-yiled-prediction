from __future__ import annotations

from pathlib import Path
import pandas as pd

KEYS = ["state", "district", "crop", "season", "year"]
BASE_NUMERIC = ["previous_yield"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # The public ICRISAT/NASA/FAO starter panel stores yield directly in kg/ha.
    if {"Year", "Region", "Crop", "Season", "Yield"}.issubset(df.columns):
        parts = df["Region"].astype("string").str.rsplit(",", n=1, expand=True)
        if parts.shape[1] != 2:
            raise ValueError("Expected Region values formatted as 'District, State'.")
        df = df.rename(columns={"Year":"year", "Crop":"crop", "Season":"season"})
        df["district"], df["state"] = parts[0].str.strip(), parts[1].str.strip()
        df["yield_t_ha"] = pd.to_numeric(df["Yield"], errors="coerce") / 1000.0
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["crop"] = df["crop"].astype("string").str.strip()
        df["season"] = df["season"].astype("string").str.strip()
        df = df.dropna(subset=KEYS + ["yield_t_ha"])
        df = df[df["yield_t_ha"] >= 0].copy()
        if df.duplicated(KEYS).any():
            raise ValueError("Duplicate district/crop/season/year keys in source dataset.")
        return df.sort_values(KEYS).reset_index(drop=True)
    if set(KEYS + ["yield_t_ha"]).issubset(df.columns):
        for col in ["state", "district", "crop", "season"]:
            df[col] = df[col].astype("string").str.strip()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["yield_t_ha"] = pd.to_numeric(df["yield_t_ha"], errors="coerce")
        df = df.dropna(subset=KEYS + ["yield_t_ha"])
        if df.duplicated(KEYS).any():
            raise ValueError("Duplicate district/crop/season/year keys in prepared data.")
        return df.sort_values(KEYS).reset_index(drop=True)
    required = set(KEYS + ["production_tonnes", "area_hectares"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    for col in ["state", "district", "crop", "season"]:
        df[col] = df[col].astype("string").str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in ["production_tonnes", "area_hectares"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=KEYS + ["production_tonnes", "area_hectares"])
    df = df[(df["production_tonnes"] >= 0) & (df["area_hectares"] > 0)].copy()
    df["yield_t_ha"] = df["production_tonnes"] / df["area_hectares"]
    if df.duplicated(KEYS).any():
        examples = df.loc[df.duplicated(KEYS, keep=False), KEYS].head(4).to_dict("records")
        raise ValueError(f"Duplicate observation keys found (examples: {examples}). Resolve them before training.")
    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].replace([float("inf"), -float("inf")], pd.NA)
    df = df.dropna(subset=["yield_t_ha"])
    return df.sort_values(KEYS).reset_index(drop=True)


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(KEYS).copy()
    group = ["state", "district", "crop", "season"]
    out["previous_yield"] = out.groupby(group, dropna=False)["yield_t_ha"].shift(1)
    out["previous_3yr_mean"] = out.groupby(group, dropna=False)["yield_t_ha"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    return out


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    cats = ["state", "district", "crop", "season"]
    base = [c for c in cats if c in df.columns] + [c for c in ["year", "previous_yield", "previous_3yr_mean"] if c in df.columns]
    weather_aliases = {"rainfall_mm", "rainfall", "temperature_c", "temperature", "temp_mean_c", "humidity_pct", "humidity"}
    weather = [c for c in df.columns if c.lower() in weather_aliases]
    soil_aliases = {"soil type", "pH", "nitrogen", "phosphorus", "potassium", "fertilizer", "pesticide"}
    soils = [c for c in df.columns if c.lower().startswith("soil_") or c.lower() in {x.lower() for x in soil_aliases}]
    return {"history": base, "history_weather": base + weather, "history_weather_soil": base + weather + soils}
