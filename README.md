# Sports Car Price Prediction

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train model and save outputs

```bash
python train_model.py
```

This will create:

| Output | Description |
|--------|-------------|
| **model_artifacts/model.joblib** | Trained Random Forest model |
| **model_artifacts/scaler.joblib** | Feature scaler |
| **model_artifacts/label_encoder_make.joblib** | Brand encoder |
| **model_artifacts/feature_cols.joblib** | Feature names |
| **model_artifacts/test_predictions.csv** | Test set with actual vs predicted prices and difference |
| **model_artifacts/model_evaluation_metrics.json** | Train/test MAE, RMSE, R², MAPE (separate file) |
| **model_artifacts/model_evaluation_metrics.csv** | Same metrics in CSV (separate file) |
| **model_artifacts/visualizations/** | All plots saved as PNG files (separate files): |
| → 01_actual_vs_predicted_scatter.png | Scatter: actual vs predicted |
| → 02_actual_vs_predicted_line.png | Line chart: first 50 cars |
| → 03_error_by_brand_bar.png | Bar: mean error by brand |
| → 04_prediction_difference_bars.png | Bar: over/under per car |
| → 05_residuals_histogram.png | Histogram of errors |

## 2. Run the interface

```bash
streamlit run app.py
```

The app shows model accuracy, prediction differences, performance metrics (loaded from the saved JSON/CSV), and the saved visualizations.
