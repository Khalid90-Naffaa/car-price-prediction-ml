"""
Sports Car Price Prediction — Streamlit interface.
Uses saved model, predictions, evaluation metrics, and visualization files.
Run: streamlit run app.py
"""
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ARTIFACTS = Path(__file__).parent / "model_artifacts"
VIZ_DIR = ARTIFACTS / "visualizations"
METRICS_JSON = ARTIFACTS / "model_evaluation_metrics.json"
METRICS_CSV = ARTIFACTS / "model_evaluation_metrics.csv"
PREDICTIONS_CSV = ARTIFACTS / "test_predictions.csv"

st.set_page_config(page_title="Sports Car Price Model", layout="wide")
st.title("Sports Car Price Prediction — Model Performance")

if not (ARTIFACTS / "model.joblib").exists():
    st.error("Model not found. Run: python train_model.py")
    st.stop()


@st.cache_data
def load_metrics_from_file():
    if METRICS_JSON.exists():
        with open(METRICS_JSON) as f:
            return json.load(f)
    if (ARTIFACTS / "metrics.joblib").exists():
        m = joblib.load(ARTIFACTS / "metrics.joblib")
        return {"train": m["train"], "test": {**m["test"], "mape_percent": None}}
    return None


@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_CSV)


metrics = load_metrics_from_file()
test_df = load_predictions()

if metrics is None:
    st.error("Evaluation metrics file not found. Run: python train_model.py")
    st.stop()

# ---- Performance metrics (from saved file) ----
st.header("Model accuracy & performance metrics")
st.caption("Loaded from: model_evaluation_metrics.json")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Test R² (accuracy)", f"{metrics['test']['r2']:.4f}")
with col2:
    st.metric("Test MAE (USD)", f"${metrics['test']['mae']:,.0f}")
with col3:
    st.metric("Test RMSE (USD)", f"${metrics['test']['rmse']:,.0f}")
with col4:
    mape = metrics["test"].get("mape_percent")
    if mape is not None:
        st.metric("Mean Abs % Error", f"{mape:.2f}%")
    else:
        mape = (np.abs(test_df["Difference"]) / test_df["Actual_Price"]).mean() * 100
        st.metric("Mean Abs % Error", f"{mape:.2f}%")

st.subheader("Train vs test metrics (from saved metrics file)")
comparison = pd.DataFrame({
    "Split": ["Train", "Test"],
    "MAE (USD)": [metrics["train"]["mae"], metrics["test"]["mae"]],
    "RMSE (USD)": [metrics["train"]["rmse"], metrics["test"]["rmse"]],
    "R²": [metrics["train"]["r2"], metrics["test"]["r2"]],
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

if METRICS_CSV.exists():
    with st.expander("View raw metrics file (model_evaluation_metrics.csv)"):
        st.dataframe(pd.read_csv(METRICS_CSV), use_container_width=True, hide_index=True)

# ---- Visualizations (from saved files) ----
st.header("Saved visualizations")

viz_files = [
    ("01_actual_vs_predicted_scatter.png", "Actual vs predicted prices (scatter)"),
    ("02_actual_vs_predicted_line.png", "Actual vs predicted — line comparison (first 50)"),
    ("03_error_by_brand_bar.png", "Average prediction error by brand"),
    ("04_prediction_difference_bars.png", "Prediction difference per car (first 40)"),
    ("05_residuals_histogram.png", "Distribution of prediction errors"),
]

for filename, title in viz_files:
    path = VIZ_DIR / filename
    if path.exists():
        st.subheader(title)
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Saved image not found: {filename}. Run python train_model.py to generate it.")

# ---- Table of predictions (from saved file) ----
st.header("Test set: actual vs predicted and difference")
st.caption("Loaded from: test_predictions.csv")

display_cols = ["Car Make", "Car Model", "Year", "Actual_Price", "Predicted_Price", "Difference"]
df_display = test_df[display_cols].rename(columns={
    "Actual_Price": "Actual (USD)",
    "Predicted_Price": "Predicted (USD)",
    "Difference": "Difference (USD)",
})
st.dataframe(
    df_display.style.format({
        "Actual (USD)": "${:,.0f}",
        "Predicted (USD)": "${:,.0f}",
        "Difference (USD)": "${:,.0f}",
    }),
    use_container_width=True,
    hide_index=True,
)
st.caption("Difference = Predicted − Actual. Positive = overestimated; negative = underestimated.")
