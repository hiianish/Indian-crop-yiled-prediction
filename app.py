from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from src.data import load_dataset, add_history

st.set_page_config(page_title="Indian Crop Yield Explorer", page_icon="🌾", layout="wide")
st.title("🌾 Indian Crop Yield Prediction")
st.caption("Compare historical yields and explore trained model predictions. Results depend on the dataset you provide.")

uploaded = st.sidebar.file_uploader("Upload prepared crop_yield.csv", type="csv")
path = Path("data/processed/crop_yield.csv")
try:
    data = load_dataset(uploaded if uploaded else path) if uploaded else load_dataset(path)
except FileNotFoundError:
    st.info("Add a prepared dataset at `data/processed/crop_yield.csv` or upload one in the sidebar. See `data/README.md` for its schema.")
    st.stop()
except Exception as exc:
    st.error(str(exc)); st.stop()

data = add_history(data)
crop = st.sidebar.selectbox("Crop", sorted(data.crop.dropna().unique()))
subset = data[data.crop == crop]
col1,col2,col3 = st.columns(3)
col1.metric("Observations", f"{len(subset):,}")
col2.metric("Median yield", f"{subset.yield_t_ha.median():.2f} t/ha")
col3.metric("Years", f"{int(subset.year.min())}–{int(subset.year.max())}")
st.subheader(f"{crop}: yield by year")
st.line_chart(subset.groupby("year").yield_t_ha.median())
st.subheader("Regional observations")
st.dataframe(subset.groupby(["state","district","season"], as_index=False).agg(observations=("yield_t_ha","size"), median_yield_t_ha=("yield_t_ha","median")).sort_values("median_yield_t_ha", ascending=False), use_container_width=True)

metric_path=Path("artifacts/metrics.csv")
if metric_path.exists():
    metrics=pd.read_csv(metric_path)
    st.subheader("Held-out model comparison")
    st.caption("Chronological test-set results from the latest training run.")
    st.dataframe(metrics.sort_values("rmse"), use_container_width=True)
    st.bar_chart(metrics.pivot(index="model", columns="feature_set", values="rmse"))

st.subheader("Predict a yield")
model_files=sorted(Path("artifacts").glob("*.joblib"))
if not model_files:
    st.caption("Train models with `python -m src.train` to enable predictions.")
else:
    bundle_path=st.selectbox("Saved model", model_files, format_func=lambda p:p.stem.replace("_"," ").title())
    bundle=joblib.load(bundle_path)
    features=bundle["features"]
    row={}
    for c in features:
        if c in {"state","district","crop","season"}:
            choices=sorted(data[c].dropna().astype(str).unique())
            row[c]=st.selectbox(c.title(), choices) if choices else ""
        else:
            default=float(data[c].median()) if c in data and data[c].notna().any() else 0.0
            row[c]=st.number_input(c.replace("_"," ").title(), value=default)
    if st.button("Predict yield", type="primary"):
        prediction=float(bundle["pipeline"].predict(pd.DataFrame([row], columns=features))[0])
        st.success(f"Predicted yield: **{prediction:.3f} tonnes/hectare** · {bundle['model_name']}")
