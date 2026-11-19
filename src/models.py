from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


def estimators(seed: int = 42):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=3, random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=30, max_depth=12, min_samples_leaf=2, n_jobs=1, random_state=seed),
        "SVR": SVR(C=10, epsilon=0.1),
        "MLP": MLPRegressor(hidden_layer_sizes=(48, 24), early_stopping=True, max_iter=100, random_state=seed),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=80, max_depth=6, learning_rate=.05, subsample=.85, colsample_bytree=.85, objective="reg:squarederror", n_jobs=1, random_state=seed)
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(iterations=80, depth=7, learning_rate=.05, verbose=False, random_seed=seed, allow_writing_files=False, thread_count=1)
    except ImportError:
        pass
    return models


def make_pipeline(df, columns, estimator):
    categorical = [c for c in columns if c in df.columns and (not pd.api.types.is_numeric_dtype(df[c]) or c in {"state", "district", "crop", "season"})]
    numeric = [c for c in columns if c not in categorical]
    prep = ColumnTransformer([
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ], remainder="drop")
    return Pipeline([("preprocess", prep), ("model", estimator)])
